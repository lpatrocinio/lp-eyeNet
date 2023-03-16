import numpy as np
from tensorflow import image
import random
from skimage import transform

def random_flip(img, label, u=0.5):
    
    if np.random.random() < u:
        img = np.fliplr(img)
        label = np.fliplr(label)
        
    if np.random.random() < u:
        img = np.flipud(img)
        label = np.flipud(label)
        
    return img, label


def random_rotate(img, label, max_angle, u=0.5):
    
    angle = random.uniform(-max_angle, max_angle)
    if np.random.random() < u:
        img = transform.rotate(img, angle, mode='edge', preserve_range=True)
        label = transform.rotate(label, angle, mode='edge', preserve_range=True)

    return img, label


def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    x = image.apply_affine_transform(x, ty=ty, tx=tx)
    return x


def random_shift(img, label, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = shift(img, wshift, hshift)
        label = shift(label, wshift, hshift)

    return img, label


def random_zoom(img, label, zoom_range=(0.8, 1), u=0.5):
    
    if np.random.random() < u:
        
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = image.apply_affine_transform(img, zx=zx, zy=zy)
        label = image.apply_affine_transform(label, zx=zx, zy=zy)
        
    return img, label


def random_shear(img, label, intensity_range=(-0.5, 0.5), u=0.5):
    
    if np.random.random() < u:
        
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        
        img = image.apply_affine_transform(img, shear=sh)
        label = image.apply_affine_transform(label, shear=sh)
        
    return img, label


def random_gray(img, label, u=0.5):
    
    if np.random.random() < u:
        
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(img * coef, axis=2)
        
        img = np.dstack((gray, gray, gray))
        
    return img


def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
        
    return img


def random_brightness(img, limit=(-0.1, 0.1), u=0.5):
    
    if np.random.random() < u:
        
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
        
    return img


def random_saturation(img, limit=(-0.1, 0.1), u=0.5):
    
    if np.random.random() < u:
        
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 1.)
        
    return img


def random_channel_shift(x, limit, channel_axis=2):
    
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    
    return x


def random_augmentation(img, label):
    
    # img, label = random_rotate(img, label, 180, u=0.5)
    img, label = random_flip(img, label, u=0.5)
    # # img, label = random_shift(img, label, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.05)
    # img = random_saturation(img)
    # img = random_brightness(img)
    return img, label