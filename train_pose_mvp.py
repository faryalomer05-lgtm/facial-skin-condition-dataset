"""Non-clinical EfficientNet-B0 capture-angle trainer for Cutilytics AI."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

POSES={"front":"front","right":"right_profile","left":"left_profile"}

def resolve(root,value):
    raw=str(value).replace('\\','/')
    direct=root/raw
    if direct.exists(): return direct
    stem=Path(raw).stem
    matches=list(root.rglob(stem+'.*'))
    return next((p for p in matches if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}),None)

def dataset(root,csv):
    meta=pd.read_csv(csv);rows=[]
    for _,row in meta.iterrows():
        for col in meta.columns:
            pose=next((v for k,v in POSES.items() if k in col.lower()),None)
            if pose:
                image=resolve(root,row[col])
                if image: rows.append({'path':str(image),'person':str(row['id']),'pose':pose})
    frame=pd.DataFrame(rows)
    if len(frame)<12 or frame.pose.nunique()<3: raise ValueError(f'Expected three poses; found {len(frame)} usable images.')
    return frame

def make_ds(frame,mapping,shuffle=False):
    ds=tf.data.Dataset.from_tensor_slices((frame.path.values,frame.pose.map(mapping).values))
    def read(path,label):
        image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
        image=tf.image.resize(image,(224,224));image.set_shape((224,224,3));return image,label
    if shuffle: ds=ds.shuffle(len(frame),seed=42)
    return ds.map(read,num_parallel_calls=tf.data.AUTOTUNE).batch(12).prefetch(tf.data.AUTOTUNE)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='.');p.add_argument('--csv',default='Facial Skin Condition Dataset.csv');p.add_argument('--output-dir',default='models');p.add_argument('--epochs',type=int,default=12);a=p.parse_args()
    root=Path(a.data_dir);out=Path(a.output_dir);out.mkdir(exist_ok=True)
    data=dataset(root,a.csv);split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    train_i,test_i=next(split.split(data,groups=data.person));train,test=data.iloc[train_i],data.iloc[test_i]
    if set(train.person)&set(test.person): raise RuntimeError('Person leakage detected')
    classes=sorted(data.pose.unique());mapping={c:i for i,c in enumerate(classes)}
    base=tf.keras.applications.EfficientNetB0(include_top=False,weights='imagenet',input_shape=(224,224,3));base.trainable=False
    inputs=tf.keras.Input((224,224,3));x=base(inputs,training=False);x=tf.keras.layers.GlobalAveragePooling2D()(x);x=tf.keras.layers.Dropout(.3)(x);outputs=tf.keras.layers.Dense(len(classes),activation='softmax')(x)
    model=tf.keras.Model(inputs,outputs);model.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(make_ds(train,mapping,True),validation_data=make_ds(test,mapping),epochs=a.epochs,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)])
    metrics=model.evaluate(make_ds(test,mapping),return_dict=True);model.save(out/'cutilytics_efficientnetb0.keras')
    (out/'class_names.json').write_text(json.dumps(classes));(out/'model_metadata.json').write_text(json.dumps({'task':'pose_recognition','split':'person-level 80/20'}));(out/'test_metrics.json').write_text(json.dumps({k:float(v) for k,v in metrics.items()},indent=2))
if __name__=='__main__':main()
