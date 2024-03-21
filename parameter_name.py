
def para_name(opt):
    if opt.modality == 'img':
        name_para = 'use_model={}~method={}~bs={}~decay={}~lr={}~lrd_rate={}'.format(

            'img',
            opt.method,
            opt.batch_size,
            opt.lr_decay,
            opt.lr,
            opt.lrd_rate
        )
    if opt.modality == 'point':
        name_para = 'use_model={}~method={}~bs={}~decay={}~lr={}~lrd_rate={}'.format(

            'point',
            opt.method,
            opt.batch_size,
            opt.lr_decay,
            opt.lr,
            opt.lrd_rate
        )
    if opt.modality == 'img_point':
        name_para = '~use_model={}~method={}~bs={}~decay={}~lr={}~lrd_rate={}'.format(

            'img_point',
            opt.method,
            opt.batch_size,
            opt.lr_decay,
            opt.lr,
            opt.lrd_rate
        )
    return name_para