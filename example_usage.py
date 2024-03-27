import pygame
import numpy as np
from dynamic_graphs_with_pygame import dynamic_pygame_graphs_class


# Initialize Pygame
pygame.init()


# Set up the display
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Randomized NumPy Array")


# Make the instance
x = 30
y = 30
pygame_graphs_instance = dynamic_pygame_graphs_class(x, y, screen)


# Main loop
running = True
while running:

    # Handle events.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen with white color.
    screen.fill((255, 255, 255))

    # Scrolling with the arrow keys.
    pygame_graphs_instance.scroll(scrolling_speed=10)

    # Generate a new randomized NumPy array.
    randomized_array = np.random.uniform(0, 230, 300)

    # Drawing the graph.
    pygame_graphs_instance.dynamic_histogram(x_values=randomized_array,
                                             bin_size=1,
                                             bar_color=(255, 0, 0),
                                             graph_x=500,
                                             graph_y=500,
                                             graph_x_axis_name='',
                                             graph_y_axis_name='',
                                             x_indicator_points=5,
                                             y_indicator_points=5,
                                             y_amplifier=1,
                                             graph_indicators_font=None,
                                             graph_indicators_text_color=(0, 0, 0),
                                             graph_indicators_text_space_from_x_axis=10,
                                             graph_indicators_text_space_from_y_axis=20,
                                             move_zero_along_x_axis=0,
                                             move_zero_along_x_axis_indicator_text_color=(0, 0, 0),
                                             bin_array_is_given_as_x_values=True,
                                             have_extra_bin=True)

    # Generate a new randomized NumPy array.
    randomized_array = np.zeros(500)
    randomized_array[200:] = np.random.uniform(0, 460, 300)

    # Drawing the second graph to show that you can make multiple and let them overlap.
    # Also shows the usefulness of alpha_coloring (look at the fourth value in the color).
    pygame_graphs_instance.dynamic_histogram(x_values=randomized_array,
                                             bin_size=10,
                                             bar_color=(0, 0, 255, 123),
                                             graph_x=500,
                                             graph_y=500,
                                             graph_x_axis_name='',
                                             graph_y_axis_name='',
                                             x_indicator_points=5,
                                             y_indicator_points=5,
                                             y_amplifier=1,
                                             graph_indicators_font=None,
                                             graph_indicators_text_color=(0, 0, 0),
                                             graph_indicators_text_space_from_x_axis=10,
                                             graph_indicators_text_space_from_y_axis=20,
                                             move_zero_along_x_axis=0,
                                             move_zero_along_x_axis_indicator_text_color=(0, 0, 0),
                                             bin_array_is_given_as_x_values=True,
                                             have_extra_bin=True)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
