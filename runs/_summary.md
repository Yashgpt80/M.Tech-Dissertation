# Run summary

```
                   run                        model  image_size  lambda_attn mask_source cam_method  epochs_run  best_epoch  best_val_acc  best_val_f1  test_acc  test_macro_f1
   S5_swin_tiny_xai_b1 swin_tiny_patch4_window7_224         224          0.1    landmark    gradcam          18          14      0.721928     0.697429  0.734466       0.716417
   S5_swin_tiny_xai_b2 swin_tiny_patch4_window7_224         224          0.1    landmark    gradcam          18          14      0.723321     0.701475  0.734466       0.709663
      S4_swin_tiny_xai swin_tiny_patch4_window7_224         224          0.5    landmark    gradcam          16          10      0.713291     0.698978  0.717200       0.709600
   S5_swin_tiny_xai_b3 swin_tiny_patch4_window7_224         224          0.1    landmark    gradcam          12           6      0.709668     0.688453  0.724157       0.706849
    swin_tiny_baseline swin_tiny_patch4_window7_224         224          NaN         NaN        NaN          30          23      0.707439     0.681982  0.722764       0.703707
       S4_resnet50_xai                     resnet50         224          0.5    landmark    gradcam          17          11      0.686542     0.673054  0.699359       0.675129
     resnet50_baseline                     resnet50         224          NaN         NaN        NaN          40          37      0.685985     0.664738  0.695458       0.663154
mini_xception_baseline                mini_xception          48          NaN         NaN        NaN          47          32      0.493452     0.466316  0.521315       0.489667
  S4_mini_xception_xai                mini_xception          48          0.5    landmark    gradcam          20          14      0.471998     0.436631  0.480635       0.447126
```
