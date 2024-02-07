# load the breast dataset and train a diffusion model on it
import torch
import matplotlib.pyplot as plt
import os
from diffusion import FourierDiffusionModel, UnconditionalScoreEstimator
from breast_utils import UNet, TimeEncoder, PositionEncoder, breast_train_loader 

device = torch.device('cuda')

# --------------------------
# define parameters
# --------------------------
verbose=True
loadPreviousWeights=False
runTraining=True
runTesting=False
runReverseProcess=True
save_interval = 10
plot_interval = 10
img_shape = 224
# --------------------------

if __name__ == '__main__':

    time_encoder = TimeEncoder(out_channels=32).to(device)
    position_encoder = PositionEncoder(out_channels=32).to(device)
    denoiser = UNet(in_channels=65, out_channels=1, num_base_filters=32).to(device)

    class ScoreEstimator(UnconditionalScoreEstimator): #modified
        def __init__(self, denoiser, time_encoder, position_encoder):
            super().__init__()
            self.denoiser = denoiser
            self.time_encoder = time_encoder
            self.position_encoder = position_encoder

        def forward(self, x_t, t):
            t_enc = self.time_encoder(t.unsqueeze(1))
            pos_enc = self.position_encoder().repeat(x_t.shape[0], 1, 1, 1)
            denoiser_input = torch.cat((x_t, t_enc, pos_enc), dim=1)
            return self.denoiser(denoiser_input)
    
    score_estimator = ScoreEstimator(denoiser, time_encoder, position_encoder).to(device)

    # now do it with torch
    def gaussian_blur_fourier_transfer_function(fwhm, size):
        """Generate a Gaussian blur transfer function with a given FWHM and size."""
        sigma = fwhm / (2.0 * torch.sqrt(2.0 * torch.log(torch.tensor(2.0))))
        xGrid = torch.linspace(-size // 2, size // 2, steps=size).to(device)
        yGrid = torch.linspace(-size // 2, size // 2, steps=size).to(device)
        xGrid, yGrid = torch.meshgrid(xGrid, yGrid)
        rGrid = torch.sqrt(xGrid**2 + yGrid**2)
        y = torch.exp(-rGrid**2 / (2 * sigma**2))
        y /= y.sum()  # Normalize
        y = torch.fft.fft2(y)
        y = torch.abs(y)
        return y
    
    fourier_transfer_function_LPF = gaussian_blur_fourier_transfer_function(6.0, img_shape)
    fourier_transfer_function_BPF = gaussian_blur_fourier_transfer_function(3.0, img_shape) - fourier_transfer_function_LPF
    fourier_transfer_function_HPF = torch.ones(img_shape, img_shape).to(device) - fourier_transfer_function_BPF - fourier_transfer_function_LPF
    
    fourier_transfer_function_LPF = fourier_transfer_function_LPF.unsqueeze(0).unsqueeze(0)
    fourier_transfer_function_BPF = fourier_transfer_function_BPF.unsqueeze(0).unsqueeze(0)
    fourier_transfer_function_HPF = fourier_transfer_function_HPF.unsqueeze(0).unsqueeze(0)

    def modulation_transfer_function_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, img_shape, img_shape)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) * torch.exp(-5*_t*_t)
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) * torch.exp(-7*_t*_t)
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) * torch.exp(-9*_t*_t)
        return LPF + BPF + HPF

    def modulation_transfer_function_derivative_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, img_shape, img_shape)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) * (-10*_t * torch.exp(-5*_t*_t))
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) * (-14*_t * torch.exp(-7*_t*_t))
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) * (-18*_t * torch.exp(-9*_t*_t))
        return LPF + BPF + HPF + 1e-10

    def noise_power_spectrum_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, img_shape, img_shape)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) * (1.0 - torch.exp(-10*_t*_t))
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) * (1.0 - torch.exp(-14*_t*_t))
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) * (1.0 - torch.exp(-18*_t*_t))
        return LPF + BPF + HPF + 1e-10

    def noise_power_spectrum_derivative_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, img_shape, img_shape)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) *  (20*_t * torch.exp(-10*_t*_t))
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) *  (28*_t * torch.exp(-14*_t*_t))
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) *  (36*_t * torch.exp(-18*_t*_t))
        return LPF + BPF + HPF + 1e-10


    # Create the FourierDiffusionModel using the functions we've defined
    diffusion_model = FourierDiffusionModel(
        score_estimator=score_estimator,
        modulation_transfer_function_func=modulation_transfer_function_func,
        noise_power_spectrum_func=noise_power_spectrum_func,
        modulation_transfer_function_derivative_func=modulation_transfer_function_derivative_func,
        noise_power_spectrum_derivative_func=noise_power_spectrum_derivative_func
    )

    if loadPreviousWeights:
        if os.path.exists('./data/weights/diffusion_fourier_unconditional.pt'):
            diffusion_model.load_state_dict(torch.load('./data/weights/diffusion_fourier_unconditional.pt'))
            print('Loaded weights from ./data/weights/diffusion_fourier_unconditional.pt')

#    def optimizer_hook(parameter) -> None:
#        optimizer_dict[parameter].step()
#        optimizer_dict[parameter].zero_grad()

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3, foreach=False)
#    optimizer_dict = {p: torch.optim.Adam([p], foreach=False) for p in diffusion_model.parameters()}

#    for p in diffusion_model.parameters():
#        p.register_post_accumulate_grad_hook(optimizer_hook)

    for epoch in range(1,101):
        # run the training loop
        if runTraining:
            for i, x_0_batch in enumerate(breast_train_loader):
                optimizer.zero_grad()
                x_0_batch = x_0_batch.to(device)
                loss = diffusion_model.compute_loss(x_0_batch, loss_name='elbo')
                loss.backward()
                optimizer.step()
                if verbose:
                    print(f'Epoch {epoch}, iteration {i+1}, loss {loss.item()}')

        # run the test loop
#        if runTesting:
#            for i, (x_0_batch, _) in enumerate(mnist_test_loader):
#                x_0_batch = x_0_batch.to(device)
#                diffusion_model.eval()
#                with torch.no_grad():
#                    loss = diffusion_model.compute_loss(x_0_batch, loss_name='elbo')
#                diffusion_model.train()
#                if verbose:
#                    print(f'Epoch {epoch}, iteration {i}, test loss {loss.item()}')
        
        if runReverseProcess and (epoch % plot_interval == 0):

            # for i, (x_0_batch, _) in enumerate(mnist_test_loader):
            #     x_0_batch = x_0_batch.to(device)
            #     break

            # inds = torch.randperm(x_0_batch.shape[0])
            # x_0_batch = x_0_batch[inds]

            # x_T = diffusion_model.sample_x_t_given_x_0_and_t(x_0_batch, t=torch.ones(x_0_batch.shape[0]).to(device))

            x_T = torch.randn((4,1,img_shape,img_shape)).to(device)

            # run the reverse process loop
            x_t_all = diffusion_model.reverse_process(
                            x_T=x_T[0:4],
                            t_stop=0.0,
                            num_steps=1024,
                            returnFullProcess=True,
                            verbose=False
                            )

            # plot the results as an animation
            from matplotlib.animation import FuncAnimation

            fig, ax = plt.subplots(2, 2)
            ims = []
            x_t_all = x_t_all.cpu().numpy()
            for i in range(4):
                im = ax[i//2, i%2].imshow(x_t_all[0,i, 0, :, :], animated=True)
                im.set_clim(-1, 1)
                ims.append([im])
            
            def updatefig(frame):
                print('Animating frame ', frame)
                for i in range(4):
                    if frame < 64:
                        ims[i][0].set_array(x_t_all[frame*16 + 15,i, 0, :, :])
                return [im[0] for im in ims]
            
            ani = FuncAnimation(fig, updatefig, frames=range(64), interval=50, blit=True)

            ani.save(f'./data/animations_train/diffusion_fourier_unconditional_{epoch}.gif')
        if epoch % save_interval == 0:
            torch.save(diffusion_model.state_dict(), f'./data/weights/diffusion_fourier_unconditional_{epoch}.pt')