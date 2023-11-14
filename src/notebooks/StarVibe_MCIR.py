#%%
import sirf.Gadgetron as pMR

import numpy as np
import matplotlib.pyplot as plt

#%%
# Trajectory for Golden angle radial acquisitions
def calc_rad_traj_golden(ad):
    dims = ad.dimensions()
    kx = np.linspace(-dims[2]//2, dims[2]//2, dims[2])
    ky = ad.get_ISMRMRD_info('kspace_encode_step_1')
    
    # Define angle with Golden angle increment
    angRad = ky * np.pi * 0.618034

    # Normalise radial points between [-0.5 0.5]
    krad = kx
    krad = krad / (np.amax(np.abs(krad)) * 2)

    # Calculate trajectory
    rad_traj = np.zeros((dims[2], dims[0], 2), dtype=np.float32)
    rad_traj[:, :, 0] = krad.reshape(-1, 1) * np.cos(angRad)
    rad_traj[:, :, 1] = krad.reshape(-1, 1) * np.sin(angRad)
    rad_traj = np.moveaxis(rad_traj, 0, 1)
    return(rad_traj)

#%%
# Read in data

filename = '/data/Paul/' + 'meas_MID00037_FID08328_Tho_starvibe_BodyCOMPASS_USER_noFS_kz38.h5'

# Nothing to ignore except for noise samples
ignored = pMR.IgnoreMask() 
acq_data = pMR.AcquisitionData(filename, False, ignored=ignored)
acq_data.sort_by_time()

# Get k-space dimensions and encoding limits
print(f'Dimensions of acq data: {acq_data.dimensions()}')

encode_step_1 = acq_data.get_ISMRMRD_info('kspace_encode_step_1')
print(f'Angle index goes from {np.min(encode_step_1)} to {np.max(encode_step_1)}')
encode_step_2 = acq_data.get_ISMRMRD_info('kspace_encode_step_2')
print(f'Slice index goes from {np.min(encode_step_2)} to {np.max(encode_step_2)}')

#%%
# Now we create the trajectory and set it
ktraj = calc_rad_traj_golden(acq_data)
pMR.set_radial2D_trajectory(acq_data, ktraj)

#%%
# Calculate the coil sensitivity maps
csm = pMR.CoilSensitivityData()
csm.smoothness = 100
csm.calculate(acq_data)
csm_arr = csm.as_array()
print(f'Shape of csm data: {csm_arr.shape}')

#%%
# Visualise coil maps
fig, ax = plt.subplots(4,5) # 4 coils, 5 slices
slice_step = csm_arr.shape[0]//ax.shape[1]
for cnd in range(ax.shape[0]):
    for snd in range(ax.shape[1]):
        ax[cnd,snd].imshow(np.abs(csm_arr[cnd, slice_step*snd,:,:]))

#%%
# Define acquisition model
E_sos = pMR.AcquisitionModel(acqs=acq_data, imgs=csm)
E_sos.set_coil_sensitivity_maps(csm)

#%%
# Reconstruct images
im_data = E_sos.inverse(acq_data)
im_arr = im_data.as_array()
print(f'Shape of image data: {im_arr.shape}')

#%%
# Visualise images
fig, ax = plt.subplots(2,5) # 2 views, 5 slices
z_step = im_arr.shape[0]//ax.shape[1]
y_step = im_arr.shape[1]//ax.shape[1]
for ind in range(ax.shape[1]):
    ax[0,ind].imshow(np.abs(im_arr[z_step*ind,:,:]))
    ax[1,ind].imshow(np.abs(im_arr[:,y_step*ind,:]))
    
plt.show()