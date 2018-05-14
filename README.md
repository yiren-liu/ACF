# ACF
Attentive collaborative filtering: Multimedia recommendation with item-and component-level attention



## Input discription

[num_sample, num_video,num_frame] 因为输入的时候是输入一个batch，所以num_sample这代表一次输入多个(i,j,k), num_video表示和这个用户有交互的所有视频，num_frame表示每个视频frame的数量



feat_idx #[num_sample, num_video] 表示一个batch的用户，每个用户交互的视频的编号

new_Vemb 

uemb_vec：隐含u

 

feat [num_sample,num_video,dim_feat] = new_Vemb :猜测表示一个batch的用户，每个用户交互的视频和每个视频的所有feature，就是一个视频的多个frame全部放在第2维度

```
c_F_values = np.asarray(
    np.random.normal(scale=0.1, size=(128,2)),
    dtype=np.float32
)
print(c_F_values.shape)
```

```
(users_b, pos_items_b, neg_items_b,new_items_u,mask,items_feature)
user_b (batch_size,)
pos_items_b (batch_size,)
neg_items_b (batch_size,)
new_items_u
mask_frame=new_items_u  (batch_size, max_items_num) #a sample is showed below
[10, 13, 28, 36, 40, 77, 86, 101, 105, 112, 124, 157, 161, 165, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175]

mask (batch_size, max_items_num) #a sample is showed below
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0.])

items_feature (batch_size, items_number, components_num, feature_size, )

```





## Dataset.py

hold_k_out表示生成测试集时，最好k个交互信息用来生成测试集合



## model.py

```
mask[num_sample, num_video]是由0和1组成，表示用户和视频是否有交互
feat_idx[num_sample, num_video]例如[[3,4,-1],[5,-1,-1],[6,2,4]]，表示某个用户和第几个视频有交互，-1表示没交互的内容（为了对齐）.例如，3表示用户1和第3个视频有交互
feat[num_sample, num_video, num_frame, num_feature]
mask_frame[num_sample, num_video, num_frame] 是由0和1组成，表示用户和视频以及frame是否有交互
self.video_features [video_num, num_frame, num_feature]
```
## New_Dataset

### input 

video_features

[user, num_video, num_frame, num_feature]

frame_mask

[num_sample, num_video, num_frame]



u_list_map

1D代表用户，2D代表用户有交互的商品



u_list_map得到一个字典, key表示要拿去训练的用户景点交互，value表示这个用户在该时间以前交互过的图片，

```python
{   
	(user,time,photo_id):[0:[(user,time,photo_id),....], 1:[(user,time,photo_id),....],...], 
     ...
}
```

得到一个mask字典，当时间段在time之前且有发过照片就为1

```
{
    (user,time,photo_id):[0,1,1,0,0,0,1,0,1,0],
    .....
}
```

再得到一个  

```
{
}
```

