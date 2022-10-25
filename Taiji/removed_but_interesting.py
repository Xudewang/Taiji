def propagate_err_mu(intens, intens_err, zpt0, pix=0.259, exp_time=1):
    ''' 
    How to propagate the error.
    
    ellipse_data: the data array from Iraf ellipse task.
    
    zpt0: zero point of the magnitude
    
    pix: pixel scale
    
    exp_time: exposure time
    
    '''
    texp = exp_time
    A = pix**2

    #     intens = ellipse_data['intens']
    #     intens_err = ellipse_data['int_err']

    #     intens_err_removeindef = removeellipseIndef(intens_err)

    #     intens_err[intens_err=='INDEF'] = np.nan

    #     intens_err_removeindef = [float(intens_err[i]) for i in range(len(intens_err))]

    uncertainty_inten = unumpy.uarray(list(intens), list(intens_err))

    uncertainty_mu = -2.5 * unumpy.log10(uncertainty_inten / (texp * A)) + zpt0
    uncertainty_mu_value = unumpy.nominal_values(uncertainty_mu)
    uncertainty_mu_std = unumpy.std_devs(uncertainty_mu)

    return uncertainty_mu_value, uncertainty_mu_std