"""
Author: Christos Botos
LinkedIn: https://www.linkedin.com/in/christos-botos-2369hcty3396
Email: hcty02@gmail.com
GitHub: https://github.com/ChrisBotos

Project Date: Autumn 2023

Description:
This is the file containing a class with functions for drawing dynamic graphs on pygame.

Contact:
Feel free to contact me for any questions.
"""


import pygame
import numpy as np


class dynamic_pygame_graphs_class:
    def __init__(self, x, y, screen):
        self.reset(x, y, screen)

    # The reset function can be used to reset the instance to its initial parameters.
    def reset(self, x, y, screen):
        # The x and y indicate the top left position of the graph on the screen.
        self.x = x
        self.y = y

        # This is the pygame screen where the graph will be displayed on.
        self.screen = screen

    # Press the arrow keys to scroll on screen, feel free to change this function if you want to scroll differently.
    def scroll(self, scrolling_speed=10):
        key = pygame.key.get_pressed()
        if key[pygame.K_UP]:
            self.y += scrolling_speed
        if key[pygame.K_DOWN]:
            self.y -= scrolling_speed
        if key[pygame.K_RIGHT]:
            self.x -= scrolling_speed
        if key[pygame.K_LEFT]:
            self.x += scrolling_speed


    def dynamic_histogram(self,
                          x_values=np.array([]),
                          bin_size=1,
                          bar_color=(255, 0, 0, 123),
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
                          have_extra_bin=True):


        """Draws a dynamic histogram graph using Pygame.

        Parameters:
        - x_values: A list or array of x-values. The x_values can be a numpy array or a python list.
        - bin_size: The size of each bin in the histogram.
        - bar_color: The color of the histogram bars. The fourth number in the bar color is the alpha, allowing for transparent objects.
        - graph_x: The width of the graph area.
        - graph_y: The height of the graph area.
        - graph_x_axis_name: The label for the x-axis.
        - graph_y_axis_name: The label for the y-axis.
        - x_indicator_points: The number of x-axis indicator points.
        - y_indicator_points: The number of y-axis indicator points.
        - y_amplifier: The amplification factor for y-values.
        - graph_indicators_font: The font for graph indicators.
        - graph_indicators_text_color: The color of graph indicator text.
        - graph_indicators_text_space_from_x_axis: Space between text and x-axis.
        - graph_indicators_text_space_from_y_axis: Space between text and y-axis.
        - move_zero_along_x_axis: The number of bins to move the (y==0 & x==0) point to the right or to the left.
        - bin_array_is_given_as_x_values: A boolean indicating whether bin_array is given as x_values.
        - have_extra_bin: A boolean indicating whether to include an extra bin when x_values are not perfectly divided into bins.
        """


        """Comments about specific perhaps confusing parameters"""
        # The x_values can be a numpy array or a python list.
        #
        # The fourth number in the bar color is the alpha.
        # It allows for transparent objects.
        # Look at the draw_rect_alpha function in this class for a clever way of drawing semi_transparent objects in pygame.
        #
        # The indicator_points parameters account for the amount of indicators along a specific axis.
        #
        # The y_amplifier is a variable that is multiplied with all the y values.
        #
        # The move_zero_along_x_axis parameter allows you to move the (y==0 & x==0) point to the right or to the left.
        # It is measured in bins, its value showing the start of the bin that should start from 0 to bin_size.
        #
        # The extra bin accounts for the case where the x_values are not perfectly divided into the bins.
        # For example: raw_x_values = [1,3,0,4,1] with bin_size = 2.
        # The resulting bin_array would be [4,4] ,in case have_extra_bin is False.
        # The resulting bin_array would be [4,4,1] ,in case have_extra_bin is True.


        x_values = np.array(x_values)  # Accounting for the case of a python list.

        if graph_indicators_font is None:
            graph_indicators_font = pygame.font.SysFont("Helvetica", 10)
            font_size = 10
        else:
            font_size = graph_indicators_font.get_height()

        """Draw the graph axes"""
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y), (self.x, self.y + graph_y))
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y + graph_y), (self.x + graph_x, self.y + graph_y))

        # Draw the x_axis name
        img = graph_indicators_font.render(graph_x_axis_name, True, graph_indicators_text_color)
        self.screen.blit(img, (self.x + graph_x + graph_indicators_text_space_from_x_axis * 2,
                               self.y + graph_y + graph_indicators_text_space_from_x_axis * 2))

        # Draw the y_axis name
        img = graph_indicators_font.render(graph_y_axis_name, True, graph_indicators_text_color)
        self.screen.blit(img, (self.x - graph_indicators_text_space_from_y_axis * 2,
                               self.y - graph_indicators_text_space_from_y_axis * 2))


        """Returning, if the size of the x_values array is 0"""
        if x_values.size < 1:
            return


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
        # x_resized helps us make the values of the indicators the actual values.
        if graph_bin_size < 1:
            graph_bin_size = 1


        """Draw the rects for the bins"""
        counter = 0
        for bin_value in bin_array:

            if bin_value != 0:
                bar_coordinates = (self.x + counter * graph_bin_size,
                                   self.y + graph_y - bin_value * y_amplifier,
                                   graph_bin_size,
                                   bin_value * abs(y_amplifier))

                # I use semi_transparency in case we want to show one graph behind the other.
                # This function draws objects in pygame that can also be transparent.
                self.draw_rect_alpha(self.screen,
                                     bar_color,
                                     bar_coordinates)

            counter += 1


        """Draw the indicator_points"""
        # The move_zero_along_x_axis is measured in bins so this is how we extract the actual x value from it.
        move_zero_along_x_axis_in_actual_value = move_zero_along_x_axis * bin_size

        # This shows how many bins or how many bars are displayed between two indicator points.
        num_of_bins_between_two_x_indicator_points = int(graph_x / x_indicator_points / graph_bin_size)

        # This shows how many actual values of x exist between two indicator_points.
        distance_between_two_x_indicator_points = num_of_bins_between_two_x_indicator_points * bin_size

        # This shows how many actual values of y exist between two indicator_points.
        distance_between_two_y_indicator_points = graph_y / y_indicator_points / y_amplifier


        self.draw_indicators(graph_x,
                             graph_y,
                             x_indicator_points,
                             y_indicator_points,
                             graph_indicators_font,
                             distance_between_two_x_indicator_points,
                             distance_between_two_y_indicator_points,
                             graph_indicators_text_color,
                             graph_indicators_text_space_from_x_axis,
                             graph_indicators_text_space_from_y_axis,
                             font_size,
                             move_zero_along_x_axis_in_actual_value,
                             move_zero_along_x_axis_indicator_text_color)





    def dynamic_line_graph(self,
                           x_values=np.array([]),
                           y_values=np.array([]),
                           line_color=(255, 0, 0, 123),
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
                           graph_indicators_text_space_from_y_axis=20):

        """Draws a dynamic line graph using Pygame.

        Parameters:
        - x_values: A list or array of x-values.
        - y_values: A list or array of y-values.
        - line_color: The color of the line graph.
        - graph_x: The width of the graph area.
        - graph_y: The height of the graph area.
        - graph_x_axis_name: The label for the x-axis.
        - graph_y_axis_name: The label for the y-axis.
        - x_indicator_points: The number of x-axis indicator points.
        - y_indicator_points: The number of y-axis indicator points.
        - y_amplifier: The amplification factor for y-values.
        - graph_indicators_font: The font for graph indicators.
        - graph_indicators_text_color: The color of graph indicator text.
        - graph_indicators_text_space_from_x_axis: Space between text and x-axis.
        - graph_indicators_text_space_from_y_axis: Space between text and y-axis.
        """

        x_values = np.array(x_values)  # Accounting for the case of a python list.
        y_values = np.array(y_values)  # Accounting for the case of a python list.

        if graph_indicators_font is None:
            graph_indicators_font = pygame.font.SysFont("Helvetica", 10)
            font_size = 10
        else:
            font_size = graph_indicators_font.get_height()


        """Draw the graph axes"""
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y), (self.x, self.y + graph_y))
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y + graph_y), (self.x + graph_x, self.y + graph_y))

        # Draw the x_axis name
        img = graph_indicators_font.render(graph_x_axis_name, True, graph_indicators_text_color)
        self.screen.blit(img, (self.x - graph_indicators_text_space_from_x_axis * 2,
                               self.y - graph_indicators_text_space_from_x_axis * 2))

        # Draw the y_axis name
        img = graph_indicators_font.render(graph_y_axis_name, True, graph_indicators_text_color)
        self.screen.blit(img, (self.x + graph_x + graph_indicators_text_space_from_x_axis * 2,
                               self.y + graph_y + graph_indicators_text_space_from_x_axis * 2))


        """Returning, if the size of the x_values array is 0"""
        if x_values.size < 1 or y_values.size < 1:
            return


        """Normalize x and y values"""
        max_x = np.max(x_values)
        max_y = np.max(y_values)
        normalized_x_values = (x_values / max_x) * graph_x
        normalized_y_values = (y_values / max_y) * graph_y


        """Calculate the actual x and y values on screen"""
        on_screen_x_values = self.x + normalized_x_values
        on_screen_y_values = self.y + graph_y - normalized_y_values


        """Draw line segments"""
        for index in range(len(on_screen_x_values) - 1):
            pygame.draw.line(self.screen,
                             line_color,
                             (on_screen_x_values[index], on_screen_y_values[index]),
                             (on_screen_x_values[index + 1], on_screen_y_values[index + 1])
                             )


        """Draw the indicator_points"""
        # This shows how many actual values of x and y exist between two indicator_points.
        distance_between_two_x_indicator_points = max_x // x_indicator_points
        distance_between_two_y_indicator_points = max_y // y_indicator_points / y_amplifier


        self.draw_indicators(graph_x,
                             graph_y,
                             x_indicator_points,
                             y_indicator_points,
                             graph_indicators_font,
                             distance_between_two_x_indicator_points,
                             distance_between_two_y_indicator_points,
                             graph_indicators_text_color,
                             graph_indicators_text_space_from_x_axis,
                             graph_indicators_text_space_from_y_axis,
                             font_size,
                             move_zero_along_x_axis_in_actual_value=0)


    def draw_graph_text(self,
                        text_to_be_written='',
                        text_space_from_x_axis=500,
                        text_space_from_y_axis=25,
                        font=None,
                        text_color=(255, 255, 255)):


        if font is None:
            font = pygame.font.SysFont("Helvetica", 10)

        img = font.render(text_to_be_written, True, text_color)
        self.screen.blit(img, (self.x - text_space_from_x_axis, self.y + text_space_from_y_axis))


    def draw_rect_alpha(self, surface, color, coordinates):  # Drawn a transparent rect
        shape_surf = pygame.Surface(pygame.Rect(coordinates).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
        surface.blit(shape_surf, coordinates)


    def draw_indicators(self,
                        graph_x,
                        graph_y,
                        x_indicator_points,
                        y_indicator_points,
                        graph_indicators_font,
                        distance_between_two_x_indicator_points,
                        distance_between_two_y_indicator_points,
                        graph_indicators_text_color,
                        graph_indicators_text_space_from_x_axis,
                        graph_indicators_text_space_from_y_axis,
                        font_size,
                        move_zero_along_x_axis_in_actual_value=0,
                        move_zero_along_x_axis_indicator_text_color=(255, 0, 0)):


        """Draw the x_indicator_points"""
        # The indicator position is indicates how far right from the start of the x_axis the indicator is placed.
        x_indicator_position = self.x + graph_x // x_indicator_points

        for counter in range(1, x_indicator_points + 1):
            # We draw a line with a height of 2  (from -1 to 1 as seen below) which is the indicator.
            pygame.draw.line(self.screen, (0, 0, 0), (x_indicator_position, self.y + graph_y + 1), (x_indicator_position, self.y + graph_y - 1))

            # We draw the values of the indicators.
            img = graph_indicators_font.render(str(int(distance_between_two_x_indicator_points * counter - move_zero_along_x_axis_in_actual_value)), True, graph_indicators_text_color)
            self.screen.blit(img, (x_indicator_position + 1, self.y + graph_y + graph_indicators_text_space_from_x_axis))

            # We go to the next indicator.
            x_indicator_position += graph_x // x_indicator_points

        # This is to draw the zero.
        img = graph_indicators_font.render("0", True, move_zero_along_x_axis_indicator_text_color)
        self.screen.blit(img, (self.x + move_zero_along_x_axis_in_actual_value, self.y + graph_y + graph_indicators_text_space_from_x_axis))


        """Draw the y_indicator_points"""
        y_indicator_position = self.y + graph_y - graph_y // y_indicator_points

        for counter in range(1, y_indicator_points + 1):
            # We draw a line with a height of 2  (from -1 to 1 as seen below) which is the indicator.
            pygame.draw.line(self.screen, (0, 0, 0), (self.x + 1, self.y + y_indicator_position), (self.x - 1, self.y + y_indicator_position))

            # We draw the values of the indicators.
            img = graph_indicators_font.render(str(int(distance_between_two_y_indicator_points * counter)), True, graph_indicators_text_color)
            self.screen.blit(img, (self.x - graph_indicators_text_space_from_y_axis, y_indicator_position - font_size // 2))

            # We go to the next indicator.
            y_indicator_position -= graph_y // y_indicator_points

