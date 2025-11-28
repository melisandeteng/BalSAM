_base_ = ['_base_/rsprompter_anchor.py']
 
default_scope = 'mmdet'
custom_imports = dict(imports=['mmdet.rsprompter'], allow_failed_imports=False)

work_dir = '/ROOT_WORK_DIR/RSPrompter_bci-dsm/'
crop_size = (1024, 1024)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best='coco/segm_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=1, score_thr=0.3)
)

test_out_dir="/ROOT_WORK_DIR/RSPrompter_bci-dsm/"

vis_backends = [#dict(type='LocalVisBackend'),
               dict(type='WandbVisBackend', init_kwargs=dict(project='rsprompter-bci', group='rsprompter-final', name='rsprompter-anchor-trees-dsm-final', resume="allow", id="rsprompter-anchor-trees-dsm-final",  allow_val_change=True))
                ]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')


num_classes = 31
prompt_shape = (70, 5)  # (per img pointset, per pointset point)

#### should be changed when using different pretrain model

hf_sam_pretrain_name = "/ROOT_MODEL/sam_vit_base"
# huggingface model name, e.g. facebook/sam-vit-base
# or local repo path, e.g. work_dirs/sam_cache/sam_vit_base
hf_sam_pretrain_ckpt_path = "/ROOT_MODEL/sam_vit_base/pytorch_model.bin"
data_preprocessor = dict(
    type='DSMDetDataPreprocessor',
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], #imgnet rgb
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    dsm_norm="max",
    bgr_to_rgb=False,
    pad_mask=False, #True,
    pad_size_divisor=32,

)

# model settings
model = dict(
    type='RSPrompterAnchorDSM',
    data_preprocessor=data_preprocessor,
    decoder_freeze=False,
    shared_image_embedding=dict(
        type='RSSamPositionalEmbedding',
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        type='RSSamVisionEncoder',
        hf_pretrain_name=hf_sam_pretrain_name,
        extra_config=dict(output_hidden_states=True),
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)),
    

    neck=dict(
        feature_aggregator=dict(
            in_channels=hf_sam_pretrain_name,
            hidden_channels=32,
            select_layers=range(1, 13, 2),  #### should be changed when using different pretrain model, base: range(1, 13, 2), large: range(1, 25, 2), huge: range(1, 33, 2)
        ),
    ),

    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
    
    roi_head=dict(
        type='RSPrompterAnchorRoIPromptHeadDSM',
        bbox_head=dict(
            num_classes=num_classes,
        ),

        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        
        mask_head=dict(
            type='RSPrompterAnchorMaskHeadDSM',
            
            mask_decoder=dict(
                type='RSSamMaskDecoder',
                hf_pretrain_name=hf_sam_pretrain_name,
                init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)),
            per_pointset_point=prompt_shape[1],
            with_sincos=True,
            dsm_modulated_image = False, 
            ),
            
        )
)


dataset_type = "TreesInsSegDSMBCIDataset"
#### should be changed align with your code root and data root
code_root = '/ROOT/RSPrompter'
data_root = '' 

batch_size_per_gpu = 2
num_workers = 8
persistent_workers = True
train_pipeline = [
    dict(type='LoadImageFromFile',  channel_order="rgb",to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type = "LoadDSMFromFile", imdecode_backend ="tifffile", to_float32=True), 
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # large scale jittering
    #dict(
    #    type='RandomResize',
    #    scale=crop_size,
    #    ratio_range=(0.1, 2.0),
    #    resize_type='Resize',
    #    keep_ratio=True),
    #dict(
    #    type='RandomCrop',
    ##    crop_size=crop_size,
    #    crop_type='absolute',
    #    recompute_bbox=True,
    #    allow_negative_crop=True),
    #dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs', meta_keys = ('img_id', 'img_path', 'dsm_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction'))
]

test_pipeline = [
    dict(type='LoadImageFromFile',  channel_order="rgb", to_float32=True),
    dict(type = "LoadDSMFromFile", imdecode_backend ="tifffile", to_float32=True),
    #dict(type='Resize', scale=crop_size, keep_ratio=True),
    #dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'dsm_path', 'ori_shape', 'img_shape',
            'scale_factor')
     )
]


train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset = dict(
        type=dataset_type,
        data_root='',
        ann_file="/ROOT_DATA/BCI/BCI_2022_tilessubset_family/merged_annots_dsm_train_new.json",
        pipeline=train_pipeline,
), 

    )


val_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root="",
        ann_file="/ROOT_DATA/BCI/BCI_2022_tilessubset_family/merged_annots_dsm_val_new.json",
        #data_prefix=dict(img='tiles/'),
        pipeline =test_pipeline
    )
)
test_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root="",
        ann_file="/ROOT_DATA/BCI/BCI_2022_tilessubset_family/merged_annots_dsm_test_new.json",
        pipeline =test_pipeline
    )
)

find_unused_parameters = True

resume = True
load_from = None

base_lr = 0.00001
max_epochs=100

train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=7871),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    )
    #dict(type="ExponentialLR", gamma=0.9)
]

backend_args=None
val_evaluator = [dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    classwise=True, 
    format_only=False,
    backend_args=backend_args,
    single_class=False
), 
dict(
    type='CocoMetric',
    metric=[ 'segm'],
    classwise=False, 
    format_only=False,
    backend_args=backend_args,
    single_class=True
),
#    dict(
#    type='CocoMetric',
#    metric=['segm'],
#    classwise=True, 
#    format_only=False,
#    backend_args=backend_args,
#    single_class=False,
#    weighted=True
#)
]

test_evaluator = [
 
    dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    classwise=True, 
    format_only=False,
    backend_args=backend_args,
    single_class=False
), 
dict(
    type='CocoMetric',
    metric=[ 'segm'],
    classwise=False, 
    format_only=False,
    backend_args=backend_args,
    single_class=True
),
dict(
    type='CocoMetric',
    metric=['segm'],
    classwise=True, 
    format_only=False,
    backend_args=backend_args,
    single_class=False,
    weighted=True
)
]                       
#### AMP training config
runner_type = 'Runner'
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05)
)

randomness = dict(seed=0)

#### DeepSpeed training config
# runner_type = 'FlexibleRunner'
# strategy = dict(
#     type='DeepSpeedStrategy',
#     fp16=dict(
#         enabled=True,
#         auto_cast=False,
#         fp16_master_weights_and_grads=False,
#         loss_scale=0,
#         loss_scale_window=500,
#         hysteresis=2,
#         min_loss_scale=1,
#         initial_scale_power=15,
#     ),
#     inputs_to_half=['inputs'],
#     zero_optimization=dict(
#         stage=2,
#         allgather_partitions=True,
#         allgather_bucket_size=2e8,
#         reduce_scatter=True,
#         reduce_bucket_size='auto',
#         overlap_comm=True,
#         contiguous_gradients=True,
#     ),
# )
# optim_wrapper = dict(
#     type='DeepSpeedOptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05
#     )
# )
