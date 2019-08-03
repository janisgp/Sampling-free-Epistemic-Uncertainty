import numpy as np


def post_process_disparity(disp):
    """
    post processing of predicted disparity
    """
    
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def post_process_disparity_var(disp_var):
    """
    post processing of variance of predicted disparity
    """
    
    _, h, w = disp_var.shape
    l_disp_var = disp_var[0,:,:]
    r_disp_var = np.fliplr(disp_var[1,:,:])
    m_disp_var = 0.25 * (l_disp_var + r_disp_var)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return (1.0 - l_mask - r_mask)**2 * m_disp_var + r_mask**2 * l_disp_var + l_mask**2 * r_disp_var