import numpy as np
import astra
import os
import imageio
import time
import matplotlib.pyplot as plt
import scipy.io as io

for i in range(1):
    angluar_sub_sampling = 1
    voxel_per_mm = 10
    data_path='./full_clean_proj/'
    recon_path = './pre_CBCT/'

    if not os.path.exists(recon_path):
        os.makedirs(recon_path)

    t = time.time()
    print('load data', flush=True)

    theta = np.linspace(0, 2*np.pi,500)
    vecs1=io.loadmat('./vectors.mat')['vectors'] 
    # print(vecs.shape)
    # quit()
    vecs = np.zeros((500, 12))
    for a in range(500):
        vecs[a,:] = vecs1[a, :]


    # we add the info about walnut and orbit ID
    #data_path_full = os.path.join(data_path,  'Projections', 'tubeV{}'.format(orbit_id))
    data_path_full = data_path
    # projection index
    # there are in fact 1201, but the last and first one come from the same angle
    projs_idx = range(1,501, angluar_sub_sampling)
    projs_name = 'scan_{:06}.tif'
    #projs_rows = 576
    projs_rows = 640
    projs_cols = 640


    # create the numpy array which will receive projection data from tiff files
    projs = np.zeros((len(projs_idx), projs_rows, projs_cols), dtype=np.float32)

    trafo = lambda image : np.transpose(np.flipud(image))
    # load projection data
    for i in range(len(projs_idx)):
        print(projs_idx[i])
        a=imageio.imread(data_path_full + '%d'%projs_idx[i] + '.tif')
        projs[i]=(a-a.min())/(a.max()-a.min())

    #projs = projs[::-1,...]
    projs = np.transpose(projs, (1,0,2))
    projs = np.ascontiguousarray(projs)
    print(np.round_(time.time() - t, 3), 'sec elapsed')

    ### compute FDK reconstruction #################################################

    t = time.time();
    print('compute reconstruction', flush=True)

    # size of the reconstruction volume in voxels
    vol_sz  = 3*(44 * 10 + 8,)
    # size of a cubic voxel in mm
    vox_sz  = 1/voxel_per_mm
    # numpy array holding the reconstruction volume
    vol_rec = np.zeros(vol_sz, dtype=np.float32)

    # we need to specify the details of the reconstruction space to ASTRA
    # this is done by a "volume geometry" type of structure, in the form of a Python dictionary
    # by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
    vol_geom = astra.create_vol_geom(vol_sz)
    vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
    vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
    vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
    vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
    vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
    vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz

    # we need to specify the details of the projection space to ASTRA
    # this is done by a "projection geometry" type of structure, in the form of a Python dictionary
    proj_geom = astra.create_proj_geom('cone_vec', 640, 640, vecs)

    # register both volume and projection geometries and arrays to ASTRA
    vol_id  = astra.data3d.link('-vol', vol_geom, vol_rec)
    proj_id = astra.data3d.link('-sino', proj_geom, projs)

    # finally, create an ASTRA configuration.
    # this configuration dictionary setups an algorithm, a projection and a volume
    # geometry and returns a ASTRA algorithm, which can be run on its own
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = vol_id
    alg_id = astra.algorithm.create(cfg)

    # run FDK algorithm
    astra.algorithm.run(alg_id, 1)

    # release memory allocated by ASTRA structures
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    print(np.round_(time.time() - t, 3), 'sec elapsed')



    ### save reconstruction ########################################################

    t = time.time();
    print('save results', flush=True)

    # low level plotting
    f, ax = plt.subplots(1, 3, sharex=False, sharey=False)
    ax[0].imshow(vol_rec[vol_sz[0]//2,:,:])
    ax[1].imshow(vol_rec[:,vol_sz[1]//2,:])
    ax[2].imshow(vol_rec[:,:,vol_sz[2]//2])
    f.tight_layout()
    np.transpose(vol_rec,[0,1,2])

    for i in range(200):
        a=vol_rec[:,:,i+150]
        a=255.0*(a-a.min())/(a.max()-a.min())
        a = a.astype(int)
        imageio.imsave(recon_path+'%d'%(i+1)+'.png', a)
    print(np.round_(time.time() - t, 3), 'sec elapsed')



