# Dynamic_Graphs_with_Pygame
This is a python library created using pygame that manages the creation of graphs that update dynamically displaying the changes of x and y over time.

# How to install
This package will be added to pip so a simple "pip install dynamic_graphs_with_pygame" would suffice.
Otherwise, clone the GitHub repository.

# Possible uses
Right now the supported graphs that can be made are: histograms, line_graphs and scatter plots. 
You may use the library's logic to add your own type of graphs as well! 
It is fairly easy just look at how the existing functions were made and do something similar.
Attention! Pygame has limitations with displaying very small objects, take note of that and look at how the case of histogram bin width < 1 is handled.

Be sure to take advantage of the ability of this library to draw objects in pygame that have alpha (transparency).
This way you can display different graphs on top of each other.

# How to use
Take a look at the example_usage.py too.

Take note that all the functions exist inside a class called DynamicPygameGraphs.
Essentially you need to first make an instance of this class with details like the positions of the axes and the pygame screen you want to display it in. (your x values are allowed to be larger than the size of of the graph_x axis)

Example:
pygame_graphs_instance = DynamicPygameGraphs(x, y, screen)

Afterwards you can use this instance to draw a graph of your choosing on screen.

Example:
pygame_graphs_instance.dynamic_histogram(x_values=np.array([1, 0, 2, 3, 0, 1]),
                                         bin_size=1,
                                         bar_color=(255, 0, 0, 123),
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
                                         graph_tick_marks_text_space_from_y_axis=20,
                                         move_zero_along_x_axis=0,
                                         move_zero_along_x_axis_tick_mark_text_color=(0, 0, 0),
                                         bin_array_is_given_as_x_values=True,
                                         have_extra_bin=True)

Of course, you do not need to fill all those variables, they all have pre-given values.
As for what exactly all of them mean refer to the comments in the actual code.

Scrolling is achieved by the keyboard arrows by default. 
You may change that, look at the scroll function.

Attention: For the histogram, if you want some of your bins to be displayed as negative along the x_axis take advantage of move_zero_along_x_axis parameter to set which bin corresponds to the zero position.

Attention: If your arrays are getting ever bigger the dynamic graphs may start lagging after a while. 
A solution would be to maybe delete some elements from the arrays if their lengths pass a specific threshold.

# Was made using
Python 3.12.2
Pygame 2.5.2
numpy 1.26.4