#IF WE ARE NOT FREE FROM SIN UNTIL WE DIE, JESUS IS NOT OUR SAVIOUR, then DEATH IS - Bill Johnson 
import logging
import numpy as np
from tkinter import *
#Matplot library
from matplotlib import pyplot as plt
from tkinter import filedialog, Label, Tk
# loading Python Imaging Library
from PIL import ImageTk, Image
# To get Menu when required
from tkinter import Menu
# To get the dialog box to open when required
from tkinter import Label, Tk, filedialog
# From Messagebox
from tkinter import messagebox
# From CV2
from cv2 import cv2
# Create a window
root = Tk()

# Set Title as Image Loader
root.title("HOMOMOPHIC IMAGE FILTERING BY NEIBO AUGUSTINE")

# Set the resolution of window
root.geometry("1080x900")
# Homomorphic filter class


class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian

    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """

    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(
            I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(
            I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H=None):
        """
        Method to apply homormophic filter on an image

        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency 
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == 'butterworth':
            H = self.__butterworth_filter(
                I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'gaussian':
            H = self.__gaussian_filter(
                I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == 'external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)
# End of class HomomorphicFilter

# PART OF FUNCTIONS


def select_image():
        # grab a reference to the image panels
    global panelA, panelB

    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()
    path_out = '/home/anibe/Desktop/augustine/'
    
    img_path_in = path
    img_path_out = path_out + 'filtered.png'

    # ensure a file path was selected
    if len(path) > 0:
        img = cv2.imread(img_path_in)[:, :, 0]
        homo_filter = HomomorphicFilter(a=0.75, b=1.25)
        img_filtered = homo_filter.filter(I=img, filter_params=[30, 2])
        cv2.imwrite(img_path_out, img_filtered)
		#HISTOGRAM EQUALIZATION APPLIED HERE
        cv2.equalizeHist(img)
    # convert the images to PIL format...
    image = Image.fromarray(img)
    edged = Image.fromarray(img_filtered)
	    


    # ...and then to ImageTk format
    image = ImageTk.PhotoImage(image)
    edged = ImageTk.PhotoImage(edged)
# if the panels are None, initialize them
    if panelA is None or panelB is None:
        # the first panel will store our original image
        panelA = Label(image=image)
        panelA.image = image
        panelA.pack(side="left", padx=10, pady=10)
		

        # while the second panel will store the edge map
        panelB = Label(image=edged)
        panelB.image = edged
        panelB.pack(side="right", padx=10, pady=10)

        # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image)
        panelB.configure(image=edged)
        panelA.image = image
        panelB.image = edged


panelA = None
panelB = None

def about():

    messagebox.showinfo('About Homomophic Image Filtering System', 'Homomophic Image Filtering is developed by Neibo Augustine')

def welcome():

    messagebox.showinfo('Greeting from Neibo', 'This App is developed @Wiseplus')


#Calling Menu from Here
menu = Menu(root)
about_item = Menu(menu)
new_item = Menu(menu)

new_item.add_command(label='Select Image', command = select_image)
menu.add_cascade(label='Application Menu', menu=new_item)
#About
about_item.add_command(label='Welcome', command = welcome)
about_item.add_command(label='About', command = about)
menu.add_cascade(label='Help', menu=about_item)
root.config(menu=menu)


# kick off the GUI
root.mainloop()
