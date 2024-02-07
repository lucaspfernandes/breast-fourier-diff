from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path


def numerical_sort(value):
    # Extract the numerical part from the file name
    return int(value.split('_')[-1].split('.')[0])

def create_single_gif(input_folder, output_gif_path, frame_duration=400, text_color="black"):
    frames = []

    # Get the list of GIF files in the input folder
        # Get the list of GIF files in the input folder and sort them numerically
    gif_files = sorted([file for file in os.listdir(input_folder) if file.endswith('.gif')], key=numerical_sort)

    for gif_file in gif_files:
        gif_path = os.path.join(input_folder, gif_file)
        epoch = Path(gif_path).stem.split('_')[-1]
        # Open the GIF file
        gif = Image.open(gif_path)

        # Get the last frame
        last_frame = gif.seek(gif.n_frames - 1)

        # Create a drawing object
        draw = ImageDraw.Draw(gif)

        # Define font and position for the frame number
        font = ImageFont.load_default()  # You can customize the font if needed
        position = (10, 10)

        # Add the frame number to the last frame
        draw.text(position, f'epoch = {epoch}', fill=text_color, font=font)

        # Append the modified frame to the list of frames
        frames.append(gif.copy())

    # Save the frames as a new GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=frame_duration  # Set the duration for each frame in milliseconds
    )


if __name__ == "__main__":
    input_folder = '.\\mnist-diffusion\\data\\animations_train'
    output_gif_path = f'{input_folder}\\diffusion_fourier_unconditional_final.gif'

    # Specify frame duration in milliseconds (e.g., 200 milliseconds per frame)
    frame_duration = 600

    # Specify text color (e.g., "red", "#00FF00", etc.)
    text_color = "black"

    create_single_gif(input_folder, output_gif_path, frame_duration, text_color)