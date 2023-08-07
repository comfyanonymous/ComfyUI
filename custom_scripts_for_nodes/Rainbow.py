from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import KDTree
import colorsys
import webcolors



class extract_rainbow():

    def __init__(self):
        pass

    # Utility Functions
    def rgb_to_hsv(self,rgb):
        return colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

    def hsv_to_rgb(self,hsv):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]))

    def rgb_to_hex(self,rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def hex_to_rgb(self,hex_color):
        return webcolors.hex_to_rgb(hex_color)

    def get_most_common_colors(self,image, n_colors=4):
        image = image.resize((50, 50))    # Reduce the size
        ar = np.asarray(image)
        ar = ar.reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(ar)
        
        return [self.rgb_to_hex(tuple(map(int, center))) for center in kmeans.cluster_centers_]

    def closest_color_name(self,requested_color):
        min_colors = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        return min_colors[min(min_colors.keys())]

    def get_color_name(self,hex_color):
        try:
            closest_name = actual_name = webcolors.hex_to_name(hex_color)
        except ValueError:
            closest_name = self.closest_color_name(self.hex_to_rgb(hex_color))
            actual_name = None
        return closest_name

    def get_analogous_colors(self,hsv, angle=30):
        analogous1_hue = (hsv[0] + angle/360) % 1
        analogous2_hue = (hsv[0] - angle/360) % 1
        analogous1 = self.rgb_to_hex(self.hsv_to_rgb((analogous1_hue, hsv[1], hsv[2])))
        analogous2 = self.rgb_to_hex(self.hsv_to_rgb((analogous2_hue, hsv[1], hsv[2])))
        return analogous1, analogous2

    def get_triadic_colors(self,hsv):
        triadic1_hue = (hsv[0] + 1/3) % 1
        triadic2_hue = (hsv[0] + 2/3) % 1
        triadic1 = self.rgb_to_hex(self.hsv_to_rgb((triadic1_hue, hsv[1], hsv[2])))
        triadic2 = self.rgb_to_hex(self.hsv_to_rgb((triadic2_hue, hsv[1], hsv[2])))
        return triadic1, triadic2

    def get_complementary_color(self,hsv):
        complementary_hue = (hsv[0] + 0.5) % 1
        complementary = self.rgb_to_hex(self.hsv_to_rgb((complementary_hue, hsv[1], hsv[2])))
        return complementary

    def get_monochromatic_color(self,hsv):
        # Here we are taking 3 saturation levels for a given color
        monochromatic1 = self.rgb_to_hex(self.hsv_to_rgb((hsv[0], hsv[1]*0.3, hsv[2])))
        monochromatic2 = self.rgb_to_hex(self.hsv_to_rgb((hsv[0], hsv[1]*0.5, hsv[2])))
        monochromatic3 = self.rgb_to_hex(self.hsv_to_rgb((hsv[0], hsv[1]*0.7, hsv[2])))
        return monochromatic1, monochromatic2, monochromatic3

    def main(self,image):
        # Load image and get most common colors
        hex_colors = self.get_most_common_colors(image)

        # Get color names and HSV
        MC_hex = []
        MC_names = []
        MC_hsv = []

        for hex_color in hex_colors:
            MC_hex.append(hex_color)
            MC_names.append(self.get_color_name(hex_color))
            MC_hsv.append(self.rgb_to_hsv(self.hex_to_rgb(hex_color)))

        # If MC1 color is black or very dark, use MC2 for calculations
        dark_colors = ['black', 'darkslategrey', 'indigo', 'midnightblue', 'darkred']
        if MC_names[0].lower() in dark_colors:
            MC = MC_hex[1]
            MC_name = MC_names[1]
            MC_hsv = MC_hsv[1]
        else:
            MC = MC_hex[0]
            MC_name = MC_names[0]
            MC_hsv = MC_hsv[0]

        # Calculate color schemes
        AN1, AN2 = self.get_analogous_colors(MC_hsv)
        T1, T2 = self.get_triadic_colors(MC_hsv)
        C1 = self.get_complementary_color(MC_hsv)
        C2 = self.get_complementary_color(self.rgb_to_hsv(self.hex_to_rgb(MC_hex[1])))  # Use MC2 for second complementary color

        # Get monochromatic colors for each MC
        MO = []
        for color in MC_hex:
            hsv = self.rgb_to_hsv(self.hex_to_rgb(color))
            MO.append(self.get_monochromatic_color(hsv))

        # Generate Output
        output = f"MC1: {MC_hex[0]} - {MC_names[0]}, MC2: {MC_hex[1]} - {MC_names[1]}, MC3: {MC_hex[2]} - {MC_names[2]}, MC4: {MC_hex[3]} - {MC_names[3]}\n"
        output += f"AN1: {AN1} - {self.get_color_name(AN1)}, AN2: {AN2} - {self.get_color_name(AN2)}\n"
        output += f"T1: {T1} - {self.get_color_name(T1)}, T2: {T2} - {self.get_color_name(T2)}\n"
        output += f"C1: {C1} - {self.get_color_name(C1)}, C2: {C2} - {self.get_color_name(C2)}\n"
        output += f"MO1: {MO[0][0]} - {self.get_color_name(MO[0][0])}, MO2: {MO[1][0]} - {self.get_color_name(MO[1][0])}, MO3: {MO[2][0]} - {self.get_color_name(MO[2][0])}, MO4: {MO[3][0]} - {self.get_color_name(MO[3][0])}\n"

        # Save to a text file
        with open('color_output.txt', 'w') as f:
            f.write(output)

        return output
        
        
if __name__ == '__main__':
    rnbw = extract_rainbow()
    rnbw.main()
