cfg = dict(
    # TODO: add your source and destination folders
    #       (source is the folder in the db with the folders of all subjects)

    # full path
    source_folder=r'',
    dest_folder_nmf=r'',
    dest_folder_occlusion=r'',
    
    # only folder/file names
    nmf_basis_img_folder=r'AF-206',
    nmf_basis_img_file=r'CFD-AF-206-079-N.jpg',

    # add to the list if sm other extension is required
    valid_img_extensions=['jpg'],

    # nmf settings
    nmf_dim=5,
    nmf_max_iter=2000
)