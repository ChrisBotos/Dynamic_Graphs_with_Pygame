"""
Author: Christos Botos
GitHub: github.com/ChrisBotos
PyPi: pypi.org/user/ChrisBotos
Email: botoschristos@gmail.com
LinkedIn: www.linkedin.com/in/christos-botos-2369hcty3396

Project Starting Date: Autumn 2023

Description:
This is the file containing a class with functions for drawing dynamic graphs on pygame.

Contact:
Feel free to contact me for any questions.
"""


import pygame
import numpy as np


class DynamicPygameGraphs:
    def __init__(self, x, y, screen):
        """
        Initializes the DynamicPygameGraphs class.

        Parameters:
        - x (int): The x-coordinate of the top-left position of the graph on the screen.
        - y (int): The y-coordinate of the top-left position of the graph on the screen.
        - screen (pygame.Surface): The Pygame screen surface to draw the graph on.
        """
        self.reset(x, y, screen)


    def reset(self, x, y, screen):
        """
        Resets the instance to its initial parameters.

        Parameters:
        - x (int): The x-coordinate of the top-left position of the graph on the screen.
        - y (int): The y-coordinate of the top-left position of the graph on the screen.
        - screen (pygame.Surface): The Pygame screen surface to draw the graph on.
        """
        # The x and y indicate the top-left position of the graph on the screen.
        self.x = x
        self.y = y

        # This is the Pygame screen where the graph will be displayed.
        self.screen = screen


    """Graph Functions"""

    def dynamic_histogram(self,
                          x_values=np.array([]),
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
                          have_extra_bin=True):


        """Draws a dynamic histogram graph using Pygame.

        Parameters:
        - x_values (array-like): A list or array of x-values. The x_values can be a numpy array or a Python list.
        - bin_size (int): The size of each bin in the histogram.
        - bar_color (tuple): The color of the histogram bars in RGBA format. The fourth value is the alpha, allowing for transparency.
        - graph_x (int): The width of the graph area.
        - graph_y (int): The height of the graph area.
        - graph_x_axis_name (str): The label for the x-axis.
        - graph_y_axis_name (str): The label for the y-axis.
        - x_tick_marks (int): The number of x-axis tick marks.
        - y_tick_marks (int): The number of y-axis tick marks.
        - y_amplifier (float): The amplification factor for y-values.
        - graph_tick_marks_font (pygame.font.Font): The font for graph tick marks.
        - graph_tick_marks_text_color (tuple): The color of graph tick mark text in RGB format.
        - graph_tick_marks_text_space_from_x_axis (int): Space between text and x-axis in pixels.
        - graph_tick_marks_text_space_from_y_axis (int): Space between text and y-axis in pixels.
        - move_zero_along_x_axis (int): How much to move the (y==0 & x==0) point to the right or to the left.
        - move_zero_along_x_axis_tick_mark_text_color (tuple): The color of the x-axis tick mark text at the zero position in RGB format.
        - bin_array_is_given_as_x_values (bool): A boolean indicating whether bin_array is given as x_values.
        - have_extra_bin (bool): A boolean indicating whether to include an extra bin when x_values are not perfectly divided into bins.
        """


        """Comments about specific perhaps confusing parameters"""
        # The x_values can be a numpy array or a python list.
        #
        # The fourth number in the bar color is the alpha.
        # It allows for transparent objects.
        # Look at the draw_rect_alpha function in this class for a clever way of drawing semi_transparent objects in pygame.
        #
        # The tick_marks parameters account for the amount of tick_marks along a specific axis.
        #
        # The y_amplifier is a variable that is multiplied with all the y values.
        #
        # The move_zero_along_x_axis parameter allows you to move the (y==0 & x==0) point to the right or to the left.
        # It is measured in actual values, not bins.
        #
        # The extra bin accounts for the case where the x_values are not perfectly divided into the bins.
        # For example: raw_x_values = [1,3,0,4,1] with bin_size = 2.
        # The resulting bin_array would be [4,4] ,in case have_extra_bin is False.
        # The resulting bin_array would be [4,4,1] ,in case have_extra_bin is True.


        x_values = np.array(x_values, dtype=int)  # Accounting for the case of a python list. The dtype=int does not matter because pygame does that to float coordinates anyway.

        if graph_tick_marks_font is None:
            graph_tick_marks_font = pygame.font.SysFont("Helvetica", 10)
            font_size = 10
        else:
            font_size = graph_tick_marks_font.get_height()


        """Handling the case where the x values are raw values which have not yet been divided into the bins"""
        # Example of raw x_values: [1, 1, 2, 0, 3, 1]
        # Example of transformation into a bin_array with bin_size = 2: [1, 1, 2, 0, 3, 1] => [2, 2, 4]

        if bin_array_is_given_as_x_values:
            bin_array = x_values
            num_of_bins = len(bin_array)

        else:
            num_of_x_values = len(x_values)
            num_of_bins = num_of_x_values // bin_size  # This does not account for a possible extra bin yet, notice the //.

            # Dividing the x_values into bins.
            bin_array = np.add.reduceat(x_values, range(0, num_of_x_values, bin_size))

            if not have_extra_bin:
                # We delete the extra bin.
                bin_array = bin_array[:num_of_bins]

            else:
                num_of_bins += 1


        """Calculate the graph_bin_size"""
        # The graph_bin_size has to do with how wide the line can be in the graph.
        graph_bin_size = graph_x / num_of_bins

        # If the graph_bin_size is below 1 that is not possible for pygame. so we need to make everything under 1 to 1.
        # x_resized helps us make the values of the tick_marks the actual values.
        if graph_bin_size < 1:
            graph_bin_size = 1
            graph_x = num_of_bins


        """Draw the graph axes"""
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y), (self.x, self.y + graph_y))
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y + graph_y), (self.x + graph_x, self.y + graph_y))

        # Draw the x_axis name
        img = graph_tick_marks_font.render(graph_x_axis_name, True, graph_tick_marks_text_color)
        self.screen.blit(img, (self.x + graph_x + graph_tick_marks_text_space_from_x_axis * 2,
                               self.y + graph_y + graph_tick_marks_text_space_from_x_axis * 2))

        # Draw the y_axis name
        img = graph_tick_marks_font.render(graph_y_axis_name, True, graph_tick_marks_text_color)
        self.screen.blit(img, (self.x - graph_tick_marks_text_space_from_y_axis * 2,
                               self.y - graph_tick_marks_text_space_from_y_axis * 2))


        """Draw the rects for the bins"""
        counter = 0
        for bin_value in bin_array:

            if bin_value != 0:
                bar_height = int(bin_value * y_amplifier)

                if bar_height > 0:
                    bar_coordinates = (self.x + counter * graph_bin_size,
                                       self.y + graph_y - bar_height,
                                       graph_bin_size,
                                       bar_height)
                else:
                    bar_coordinates = (self.x + counter * graph_bin_size,
                                       self.y + graph_y,
                                       graph_bin_size,
                                       np.abs(bar_height))

                # I use semi_transparency in case we want to show one graph behind the other.
                # This function draws objects in pygame that can also be transparent.
                draw_rect_alpha(self.screen,
                                bar_color,
                                bar_coordinates)

            counter += 1


        """Draw the tick_marks"""
        # This shows how many bins or how many bars are displayed between two tick marks.
        num_of_bins_between_two_x_tick_marks = int(graph_x / x_tick_marks / graph_bin_size)

        # This shows how many actual values of x exist between two tick_marks.
        distance_between_two_x_tick_marks = num_of_bins_between_two_x_tick_marks * bin_size

        # This shows how many actual values of y exist between two tick_marks.
        distance_between_two_y_tick_marks = int(graph_y / y_tick_marks / y_amplifier)


        self.draw_tick_marks(graph_x=graph_x,
                             graph_y=graph_y,
                             x_tick_marks=x_tick_marks,
                             y_tick_marks=y_tick_marks,
                             graph_tick_marks_font=graph_tick_marks_font,
                             distance_between_two_x_tick_marks=distance_between_two_x_tick_marks,
                             distance_between_two_y_tick_marks=distance_between_two_y_tick_marks,
                             graph_tick_marks_text_color=graph_tick_marks_text_color,
                             graph_tick_marks_text_space_from_x_axis=graph_tick_marks_text_space_from_x_axis,
                             graph_tick_marks_text_space_from_y_axis=graph_tick_marks_text_space_from_y_axis,
                             font_size=font_size,
                             move_zero_along_x_axis=move_zero_along_x_axis,
                             move_zero_along_x_axis_tick_mark_text_color=move_zero_along_x_axis_tick_mark_text_color,
                             bin_size=bin_size)


    def dynamic_line_graph(self,
                           x_values=np.array([]),
                           y_values=np.array([]),
                           normalize=False,
                           line_color=(255, 0, 0, 123),
                           line_width=2,
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
                           move_zero_along_x_axis_tick_mark_text_color=(0, 0, 0)):


        """Draws a dynamic line graph using Pygame.

        Parameters:
        - x_values (array-like): A list or array of x-values.
        - y_values (array-like): A list or array of y-values.
        - normalize (bool): If set to True, normalizes the values so that the line always reaches from one end of the graph to the other.
        - line_color (tuple): The color of the line graph in RGBA format. The fourth value is the alpha, allowing for transparency.
        - line_width (int): The width of the line in the graph.
        - graph_x (int): The width of the graph area.
        - graph_y (int): The height of the graph area.
        - graph_x_axis_name (str): The label for the x-axis.
        - graph_y_axis_name (str): The label for the y-axis.
        - x_tick_marks (int): The number of x-axis tick marks.
        - y_tick_marks (int): The number of y-axis tick marks.
        - y_amplifier (float): The amplification factor for y-values.
        - graph_tick_marks_font (pygame.font.Font): The font for graph tick marks.
        - graph_tick_marks_text_color (tuple): The color of graph tick mark text in RGB format.
        - graph_tick_marks_text_space_from_x_axis (int): Space between text and x-axis in pixels.
        - graph_tick_marks_text_space_from_y_axis (int): Space between text and y-axis in pixels.
        - move_zero_along_x_axis (int): How much to move the (y==0 & x==0) point to the right or to the left.
        - move_zero_along_x_axis_tick_mark_text_color (tuple): The color of the x-axis tick mark text at the zero position in RGB format.
        """


        x_values = np.array(x_values)  # Accounting for the case of a python list.
        y_values = np.array(y_values) * y_amplifier  # Accounting for the case of a python list.

        if graph_tick_marks_font is None:
            graph_tick_marks_font = pygame.font.SysFont("Helvetica", 10)
            font_size = 10
        else:
            font_size = graph_tick_marks_font.get_height()


        """Draw the graph axes"""
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y), (self.x, self.y + graph_y))
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y + graph_y), (self.x + graph_x, self.y + graph_y))

        # Draw the x_axis name
        img = graph_tick_marks_font.render(graph_x_axis_name, True, graph_tick_marks_text_color)
        self.screen.blit(img, (self.x - graph_tick_marks_text_space_from_x_axis * 2,
                               self.y - graph_tick_marks_text_space_from_x_axis * 2))

        # Draw the y_axis name
        img = graph_tick_marks_font.render(graph_y_axis_name, True, graph_tick_marks_text_color)
        self.screen.blit(img, (self.x + graph_x + graph_tick_marks_text_space_from_x_axis * 2,
                               self.y + graph_y + graph_tick_marks_text_space_from_x_axis * 2))


        """Returning, if the size of the x_values array is 0"""
        if x_values.size < 1 or y_values.size < 1:
            return


        """Normalize x and y values"""
        max_x = np.max(x_values)
        max_y = np.max(y_values)

        if normalize:
            x_values = x_values / max_x * graph_x
            y_values = y_values / max_y * graph_y


        """Calculate the actual x and y values on screen"""
        on_screen_x_values = self.x + x_values
        on_screen_y_values = self.y + graph_y - y_values


        """Draw line segments"""
        for index in range(len(on_screen_x_values) - 1):
            pygame.draw.line(self.screen,
                             line_color,
                             (on_screen_x_values[index], on_screen_y_values[index]),
                             (on_screen_x_values[index + 1], on_screen_y_values[index + 1]),
                             line_width
                             )


        """Draw the tick_marks"""
        # This shows how many actual values of x and y exist between two tick_marks.
        if normalize:
            distance_between_two_x_tick_marks = int(max_x / x_tick_marks)
            distance_between_two_y_tick_marks = int(max_y / y_tick_marks / y_amplifier)

        else:
            distance_between_two_x_tick_marks = int(graph_x / x_tick_marks)
            distance_between_two_y_tick_marks = int(graph_y / y_tick_marks / y_amplifier)

        self.draw_tick_marks(graph_x=graph_x,
                             graph_y=graph_y,
                             x_tick_marks=x_tick_marks,
                             y_tick_marks=y_tick_marks,
                             graph_tick_marks_font=graph_tick_marks_font,
                             distance_between_two_x_tick_marks=distance_between_two_x_tick_marks,
                             distance_between_two_y_tick_marks=distance_between_two_y_tick_marks,
                             graph_tick_marks_text_color=graph_tick_marks_text_color,
                             graph_tick_marks_text_space_from_x_axis=graph_tick_marks_text_space_from_x_axis,
                             graph_tick_marks_text_space_from_y_axis=graph_tick_marks_text_space_from_y_axis,
                             font_size=font_size,
                             move_zero_along_x_axis=move_zero_along_x_axis,
                             move_zero_along_x_axis_tick_mark_text_color=move_zero_along_x_axis_tick_mark_text_color)


    def dynamic_scatter_plot(self,
                             x_values=np.array([]),
                             y_values=np.array([]),
                             shape="circle",
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
                             graph_tick_marks_text_space_from_y_axis=20,
                             move_zero_along_x_axis=0,
                             move_zero_along_x_axis_tick_mark_text_color=(0, 0, 0)):


        """Draws a dynamic scatter plot using Pygame.

        Parameters:
        - x_values (array-like): A list or array of x-values.
        - y_values (array-like): A list or array of y-values.
        - shape (str): The shape of the points. Can be 'circle' or 'square'.
        - radius (int): For circles, it is the radius; for squares, it is half the side length.
        - normalize (bool): If set to True, normalizes the values so that the points always cover the entire graph area.
        - point_color (tuple): The color of the points in RGBA format. The fourth value is the alpha, allowing for transparency.
        - graph_x (int): The width of the graph area.
        - graph_y (int): The height of the graph area.
        - graph_x_axis_name (str): The label for the x-axis.
        - graph_y_axis_name (str): The label for the y-axis.
        - x_tick_marks (int): The number of x-axis tick marks.
        - y_tick_marks (int): The number of y-axis tick marks.
        - y_amplifier (float): The amplification factor for y-values.
        - graph_tick_marks_font (pygame.font.Font): The font for graph tick marks.
        - graph_tick_marks_text_color (tuple): The color of graph tick mark text in RGB format.
        - graph_tick_marks_text_space_from_x_axis (int): Space between text and x-axis in pixels.
        - graph_tick_marks_text_space_from_y_axis (int): Space between text and y-axis in pixels.
        - move_zero_along_x_axis (int): How much to move the (y==0 & x==0) point to the right or to the left.
        - move_zero_along_x_axis_tick_mark_text_color (tuple): The color of the x-axis tick mark text at the zero position in RGB format.
        """

        if shape != "circle" and shape != "square":
            raise Exception("Only circle and square are supported as parameters for shape in the scatter plot.")

        x_values = np.array(x_values)  # Accounting for the case of a python list.
        y_values = np.array(y_values) * y_amplifier  # Accounting for the case of a python list.

        if graph_tick_marks_font is None:
            graph_tick_marks_font = pygame.font.SysFont("Helvetica", 10)
            font_size = 10
        else:
            font_size = graph_tick_marks_font.get_height()


        """Draw the graph axes"""
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y), (self.x, self.y + graph_y))
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y + graph_y), (self.x + graph_x, self.y + graph_y))

        # Draw the x_axis name
        img = graph_tick_marks_font.render(graph_x_axis_name, True, graph_tick_marks_text_color)
        self.screen.blit(img, (self.x - graph_tick_marks_text_space_from_x_axis * 2,
                               self.y - graph_tick_marks_text_space_from_x_axis * 2))

        # Draw the y_axis name
        img = graph_tick_marks_font.render(graph_y_axis_name, True, graph_tick_marks_text_color)
        self.screen.blit(img, (self.x + graph_x + graph_tick_marks_text_space_from_x_axis * 2,
                               self.y + graph_y + graph_tick_marks_text_space_from_x_axis * 2))


        """Returning, if the size of the x_values array is 0"""
        if x_values.size < 1 or y_values.size < 1:
            return


        """Normalize x and y values"""
        max_x = np.max(x_values)
        max_y = np.max(y_values)

        if normalize:
            x_values = x_values / max_x * graph_x
            y_values = y_values / max_y * graph_y


        """Calculate the actual x and y values on screen"""
        on_screen_x_values = self.x + x_values
        on_screen_y_values = self.y + graph_y - y_values


        """Draw points"""
        for index in range(len(on_screen_x_values)):

            if shape == "circle":
                draw_circle_alpha(self.screen,
                                  point_color,
                                  (on_screen_x_values[index], on_screen_y_values[index]),
                                  radius)

            elif shape == "square":
                coordinates = (on_screen_x_values[index] - radius,
                               on_screen_y_values[index] - radius,
                               radius * 2,
                               radius * 2)

                draw_rect_alpha(self.screen,
                                point_color,
                                coordinates)


        """Draw the tick_marks"""
        # This shows how many actual values of x and y exist between two tick_marks.
        if normalize:
            distance_between_two_x_tick_marks = int(max_x / x_tick_marks)
            distance_between_two_y_tick_marks = int(max_y / y_tick_marks / y_amplifier)

        else:
            distance_between_two_x_tick_marks = int(graph_x / x_tick_marks)
            distance_between_two_y_tick_marks = int(graph_y / y_tick_marks / y_amplifier)


        self.draw_tick_marks(graph_x=graph_x,
                             graph_y=graph_y,
                             x_tick_marks=x_tick_marks,
                             y_tick_marks=y_tick_marks,
                             graph_tick_marks_font=graph_tick_marks_font,
                             distance_between_two_x_tick_marks=distance_between_two_x_tick_marks,
                             distance_between_two_y_tick_marks=distance_between_two_y_tick_marks,
                             graph_tick_marks_text_color=graph_tick_marks_text_color,
                             graph_tick_marks_text_space_from_x_axis=graph_tick_marks_text_space_from_x_axis,
                             graph_tick_marks_text_space_from_y_axis=graph_tick_marks_text_space_from_y_axis,
                             font_size=font_size,
                             move_zero_along_x_axis=move_zero_along_x_axis,
                             move_zero_along_x_axis_tick_mark_text_color=move_zero_along_x_axis_tick_mark_text_color)


    """Non_Graph useful functions for the class"""

    def scroll(self, scrolling_speed=10):

        """
        Scrolls the view by adjusting the x and y coordinates of every object
        based on the current scrolling speed and the arrow key pressed.

        Parameters:
        - scrolling_speed (int, optional): The speed at which the view scrolls. Default is 10.
        """

        # Get the current state of all keyboard keys.
        key = pygame.key.get_pressed()

        # Check if the UP arrow key is pressed.
        if key[pygame.K_UP]:
            # Move all objects down by increasing their y-coordinate.
            self.y += scrolling_speed

        # Check if the DOWN arrow key is pressed.
        if key[pygame.K_DOWN]:
            # Move all objects up by decreasing their y-coordinate.
            self.y -= scrolling_speed

        # Check if the RIGHT arrow key is pressed.
        if key[pygame.K_RIGHT]:
            # Move all objects to the left by decreasing their x-coordinate.
            self.x -= scrolling_speed

        # Check if the LEFT arrow key is pressed.
        if key[pygame.K_LEFT]:
            # Move all objects to the right by increasing their x-coordinate.
            self.x += scrolling_speed


    def draw_graph_text(self,
                        text_to_be_written='',
                        text_space_from_x_axis=500,
                        text_space_from_y_axis=25,
                        font=None,
                        text_color=(255, 255, 255)):

        """
        Draws text on screen in a position relative to the graph positions.

        Parameters:
        - text_to_be_written (str): The text to be displayed.
        - text_space_from_x_axis (int): The space from the x-axis where the text will be displayed.
        - text_space_from_y_axis (int): The space from the y-axis where the text will be displayed.
        - font (pygame.font.Font, optional): The font of the text. Default is None.
        - text_color (tuple): The color of the text (RGB tuple). Default is white.
        """

        if font is None:
            font = pygame.font.SysFont("Helvetica", 10)

        img = font.render(text_to_be_written, True, text_color)
        self.screen.blit(img, (self.x - text_space_from_x_axis, self.y + text_space_from_y_axis))


    def draw_tick_marks(self,
                        graph_x,
                        graph_y,
                        x_tick_marks,
                        y_tick_marks,
                        graph_tick_marks_font,
                        distance_between_two_x_tick_marks,
                        distance_between_two_y_tick_marks,
                        graph_tick_marks_text_color,
                        graph_tick_marks_text_space_from_x_axis,
                        graph_tick_marks_text_space_from_y_axis,
                        font_size,
                        move_zero_along_x_axis=0,
                        move_zero_along_x_axis_tick_mark_text_color=None,
                        bin_size=1):

        """
        Draws the tick marks on the graph.

        Parameters:
        - graph_x (int): The width of the graph.
        - graph_y (int): The height of the graph.
        - x_tick_marks (int): The number of tick marks on the x-axis.
        - y_tick_marks (int): The number of tick marks on the y-axis.
        - graph_tick_marks_font (pygame.font.Font): The font of the tick mark text.
        - distance_between_two_x_tick_marks (int): The distance between two x-axis tick marks.
        - distance_between_two_y_tick_marks (int): The distance between two y-axis tick marks.
        - graph_tick_marks_text_color (tuple): The color of the tick mark text (RGB tuple).
        - graph_tick_marks_text_space_from_x_axis (int): The space from the x-axis where the tick mark text will be displayed.
        - graph_tick_marks_text_space_from_y_axis (int): The space from the y-axis where the tick mark text will be displayed.
        - font_size (int): The size of the font for the tick mark text.
        - move_zero_along_x_axis (int, optional): The actual value to move the zero along the x-axis. Default is 0.
        - move_zero_along_x_axis_tick_mark_text_color (tuple, optional): The color of the zero tick mark text (RGB tuple). Default is the same as graph_tick_marks_text_color.
        - bin_size (int): The size of bins for the case of histograms.
        """

        if move_zero_along_x_axis_tick_mark_text_color is None:
            move_zero_along_x_axis_tick_mark_text_color = graph_tick_marks_text_color

        # Draw the x_tick_marks
        # The tick_mark position indicates how far right from the start of the x_axis the tick_mark is placed.
        x_tick_mark_position = self.x + graph_x // x_tick_marks

        for counter in range(1, x_tick_marks + 1):
            # Draw a line with a height of 2 (from -1 to 1) which is the tick_mark.
            pygame.draw.line(self.screen, (0, 0, 0),
                             (x_tick_mark_position, self.y + graph_y + 1),
                             (x_tick_mark_position, self.y + graph_y - 1))

            # Draw the values of the tick_marks.
            img = graph_tick_marks_font.render(
                str(int(distance_between_two_x_tick_marks * counter - move_zero_along_x_axis)),
                True,
                graph_tick_marks_text_color
            )

            self.screen.blit(img,
                             (x_tick_mark_position + 1, self.y + graph_y + graph_tick_marks_text_space_from_x_axis))

            # Move to the next tick_mark.
            x_tick_mark_position += graph_x // x_tick_marks

        # Draw the zero.
        img = graph_tick_marks_font.render("0", True, move_zero_along_x_axis_tick_mark_text_color)
        self.screen.blit(img, (self.x + move_zero_along_x_axis / bin_size, self.y + graph_y + graph_tick_marks_text_space_from_x_axis))

        # Draw the y_tick_marks
        y_tick_mark_position = self.y + graph_y - graph_y // y_tick_marks

        for counter in range(1, y_tick_marks + 1):
            # Draw a line with a height of 2 (from -1 to 1) which is the tick_mark.
            pygame.draw.line(self.screen, (0, 0, 0),
                             (self.x + 1, y_tick_mark_position),
                             (self.x - 1, y_tick_mark_position))

            # Draw the values of the tick_marks.
            img = graph_tick_marks_font.render(
                str(int(distance_between_two_y_tick_marks * counter)),
                True,
                graph_tick_marks_text_color
            )
            self.screen.blit(img,
                             (self.x - graph_tick_marks_text_space_from_y_axis, y_tick_mark_position - font_size // 2))

            # Move to the next tick_mark.
            y_tick_mark_position -= graph_y // y_tick_marks


"""Functions for drawing transparent or semi-transparent shapes in pygame"""
# Explanation:
# Pygame's display surface (the screen) doesn't support per-pixel alpha blending by default.
# This means you can't directly draw transparent shapes onto the screen and expect them to blend seamlessly with the background.
# To work around this limitation, you need to create a separate surface with an alpha channel using pygame.Surface with the flag pygame.SRCALPHA.
# This creates a surface that can handle transparency.
# You then draw your transparent shape onto this surface using Pygame's drawing functions like pygame.draw.rect or pygame.draw.circle.
# Finally, you blit (copy) this transparent surface onto the screen or another surface using surface.blit.
# This allows Pygame to handle the blending of the transparent pixels with the background pixels correctly during the blitting process,
# resulting in the desired transparent effect.


"""Draw rectangles"""
def draw_rect_alpha(surface,
                    color,
                    coordinates):

    """
    Draws a transparent rectangle on a Pygame surface.

    Parameters:
        surface (pygame.Surface): Pygame surface onto which the rectangle will be drawn.
        color (tuple): Color of the rectangle (RGB or RGBA tuple).
        coordinates (tuple): Tuple representing the position and size of the rectangle (x, y, width, height).
    """

    # Create a new surface with an alpha channel (transparency)
    shape_surf = pygame.Surface(pygame.Rect(coordinates).size, pygame.SRCALPHA)

    # Draw a rectangle with the specified color on the transparent surface
    pygame.draw.rect(shape_surf, color, shape_surf.get_rect())

    # Blit (copy) the transparent surface onto the target surface at the specified coordinates
    surface.blit(shape_surf, coordinates)


"""Draw circles"""
def draw_circle_alpha(surface,
                      color,
                      center,
                      radius):

    """
    Draws a transparent circle on a Pygame surface.

    Parameters:
        surface (pygame.Surface): Pygame surface onto which the circle will be drawn.
        color (tuple): Color of the circle (RGB or RGBA tuple).
        center (tuple): Tuple representing the center coordinates of the circle (x, y).
        radius (int): Radius of the circle.
    """

    # Create a new surface with an alpha channel (transparency)
    shape_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)

    # Draw a circle with the specified color on the transparent surface
    pygame.draw.circle(shape_surf, color, (radius, radius), radius)

    # Blit (copy) the transparent surface onto the target surface with the center adjusted
    surface.blit(shape_surf, (center[0] - radius, center[1] - radius))
