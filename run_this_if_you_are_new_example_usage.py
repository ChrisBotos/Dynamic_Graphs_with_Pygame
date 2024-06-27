import pygame
import numpy as np
from dynamic_graphs_with_pygame.dyn_graphs import DynamicPygameGraphs

# Run this script and play around with the keys 1,2,3,4,space and the arrow keys on your keyboard.


# Initialize Pygame
pygame.init()


# Set up the display
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Randomized NumPy Array")


# Make the instances
x = 30
y = 30
pygame_graphs_instance1 = DynamicPygameGraphs(x, y, screen)

x = 700
y = -200
pygame_graphs_instance2 = DynamicPygameGraphs(x, y, screen)

x = 800
y = 500
pygame_graphs_instance3 = DynamicPygameGraphs(x, y, screen)

# I would recommend making a dictionary of what graphs you want to show and play around with them using keyboard keys.
what_to_show_dictionary = {
    "show_first_histogram" : True,
    "show_second_histogram" : True,
    "show_line_graph" : True,
    "show_scatter_plot" : True
}

randomized_array1 = np.random.uniform(0, 230, 500)
randomized_array2 = np.zeros(5000)
randomized_array2[2000:] = np.random.uniform(0, 80, 3000)
x_randomized_array1 = np.random.uniform(0, 600, 500)
y_randomized_array1 = np.random.uniform(0, 400, 500)
x_randomized_array2 = np.random.uniform(0, 600, 50)
y_randomized_array2 = np.random.uniform(0, 400, 50)

pause = False

# Main loop
running = True
while running:

    # Handle events.
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        # Toggle visibility of the first histogram when '1' key is pressed
        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_1:
                what_to_show_dictionary["show_first_histogram"] = not what_to_show_dictionary["show_first_histogram"]

            # Toggle visibility of the second histogram when '2' key is pressed
            if event.key == pygame.K_2:
                what_to_show_dictionary["show_second_histogram"] = not what_to_show_dictionary["show_second_histogram"]

            # Toggle visibility of the line graph when '3' key is pressed
            if event.key == pygame.K_3:
                what_to_show_dictionary["show_line_graph"] = not what_to_show_dictionary["show_line_graph"]

            # Toggle visibility of the scatter plot when '4' key is pressed
            if event.key == pygame.K_4:
                what_to_show_dictionary["show_scatter_plot"] = not what_to_show_dictionary["show_scatter_plot"]

            if event.key == pygame.K_SPACE:
                pause = not pause

    # Clear the screen with white color.
    screen.fill((255, 255, 255))

    # Scrolling with the arrow keys.
    pygame_graphs_instance1.scroll(scrolling_speed=10)
    pygame_graphs_instance2.scroll(scrolling_speed=10)
    pygame_graphs_instance3.scroll(scrolling_speed=10)


    if what_to_show_dictionary["show_first_histogram"]:
        # Generate a new randomized NumPy array.
        # This is a ready bin array! Essentially the user has already made the bin array from his original array already.
        # Notice the parameter "bin_array_is_given_as_x_values=True".

        if not pause:
            randomized_array1 = np.random.uniform(0, 230, 500)

        # Drawing the graph.
        pygame_graphs_instance1.dynamic_histogram(x_values=randomized_array1,
                                                  bin_size=1,
                                                  bar_color=(255, 0, 0),
                                                  # You do not need to make both bars alpha just one, this one is RGB not RGBA but the other below is RGBA.
                                                  graph_x=500,
                                                  graph_y=500,
                                                  graph_x_axis_name='',
                                                  graph_y_axis_name='',
                                                  x_tick_marks=5,
                                                  y_tick_marks=5,
                                                  y_amplifier=1,
                                                  graph_tick_marks_font=None,
                                                  graph_tick_marks_text_color=(255, 0, 0),
                                                  graph_tick_marks_text_space_from_x_axis=10,
                                                  graph_tick_marks_text_space_from_y_axis=20,
                                                  move_zero_along_x_axis=0,
                                                  move_zero_along_x_axis_tick_mark_text_color=(255, 0, 0),
                                                  bin_array_is_given_as_x_values=True,
                                                  have_extra_bin=True)

    if what_to_show_dictionary["show_second_histogram"]:
        # Generate a new randomized NumPy array.
        # This is a NOT ready bin array! Essentially the user has not made the bin array from his original array already.
        # Notice the parameter "bin_array_is_given_as_x_values=True".

        if not pause:
            randomized_array2 = np.zeros(5000)
            randomized_array2[2000:] = np.random.uniform(0, 80, 3000)

        # Drawing the second graph to show that you can make multiple and let them overlap.
        # Also shows the usefulness of alpha_coloring (look at the fourth value in the color).
        pygame_graphs_instance1.dynamic_histogram(x_values=randomized_array2,
                                                  bin_size=10,
                                                  bar_color=(0, 0, 255, 123),
                                                  graph_x=500,
                                                  graph_y=500,
                                                  graph_x_axis_name='',
                                                  graph_y_axis_name='',
                                                  x_tick_marks=5,
                                                  y_tick_marks=5,
                                                  y_amplifier=1,
                                                  graph_tick_marks_font=None,
                                                  graph_tick_marks_text_color=(0, 0, 255),
                                                  graph_tick_marks_text_space_from_x_axis=20,  # Notice that this is different! this way both tick mark labels will be shown when putting the graphs on top of each other.
                                                  graph_tick_marks_text_space_from_y_axis=20,
                                                  move_zero_along_x_axis=0,
                                                  move_zero_along_x_axis_tick_mark_text_color=(0, 0, 255),
                                                  bin_array_is_given_as_x_values=False,
                                                  have_extra_bin=True)
    

    if what_to_show_dictionary["show_line_graph"]:
        # Drawing a line graph a bit further away to see that you can have multiple graphs in the same screen at any coordinates you desire.
        # Move with the arrow keys to see it.
        # To draw at different coordinates you need a new instance of the class or to manually change the x and y coordinates with pygame_graphs_instance1.x = ...
        # Here I have a different instance ready.

        if not pause:
            x_randomized_array1 = np.sort(np.random.uniform(0, 600, 500))
            y_randomized_array1 = np.sort(np.random.uniform(0, 400, 500))  # Be sure to sort them if you want.

        pygame_graphs_instance2.dynamic_line_graph(x_values=x_randomized_array1,
                                                   y_values=y_randomized_array1,
                                                   normalize=True,
                                                   # Change this and see what happens, the line will go off the graph's square.
                                                   line_color=(48, 25, 52, 123),
                                                   line_width=2,
                                                   graph_x=500,
                                                   graph_y=500,
                                                   graph_x_axis_name='',
                                                   graph_y_axis_name='',
                                                   x_tick_marks=5,
                                                   y_tick_marks=5,
                                                   y_amplifier=1,  # This not being 1 is negated by normalize=True
                                                   graph_tick_marks_font=None,
                                                   graph_tick_marks_text_color=(48, 25, 52),
                                                   graph_tick_marks_text_space_from_x_axis=10,
                                                   graph_tick_marks_text_space_from_y_axis=20)


    if what_to_show_dictionary["show_scatter_plot"]:
        # Drawing a scatter plot a bit further away to see that you can have multiple graphs in the same screen at any coordinates you desire.
        # Move with the arrow keys to see it.
        # To draw at different coordinates you need a new instance of the class or to manually change the x and y coordinates with pygame_graphs_instance1.x = ...
        # Here I have a different instance ready.

        if not pause:
            x_randomized_array2 = np.random.uniform(0, 600, 50)
            y_randomized_array2 = np.random.uniform(0, 400, 50)

        pygame_graphs_instance3.dynamic_scatter_plot(x_values=x_randomized_array2,
                                                     y_values=y_randomized_array2,
                                                     shape="square",
                                                     radius=2,
                                                     normalize=False,
                                                     point_color=(255, 0, 0, 123),
                                                     graph_x=500,
                                                     graph_y=500,
                                                     graph_x_axis_name='',
                                                     graph_y_axis_name='',
                                                     x_tick_marks=5,
                                                     y_tick_marks=5,
                                                     y_amplifier=1,
                                                     graph_tick_marks_font=None,
                                                     graph_tick_marks_text_color=(0, 0, 0),
                                                     graph_tick_marks_text_space_from_x_axis=10,
                                                     graph_tick_marks_text_space_from_y_axis=20)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
