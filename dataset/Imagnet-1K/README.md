Convert ImageNet to Tfrecords
---

### Download
- [ILSVRC2012_img_train.tar](https://image-net.org/challenges/LSVRC/2012/)
- [ILSVRC2012_img_val.tar](https://image-net.org/challenges/LSVRC/2012/)
- [ILSVRC2012_bbox_train_v2.tar.gz](https://image-net.org/challenges/LSVRC/2012/)


### Bash
```bash
python preprocess_imagenet_validation_data.py /ImageNet2012/val/ \
    imagenet_2012_validation_synset_labels.txt
```

```bash    
python process_bounding_boxes.py /ImageNet2012/bbox/ \
    imagenet_lsvrc_2012_synsets.txt | sort > imagenet_2012_bounding_boxes.csv
 ```
 
 ```bash       
python build_imagenet_data.py --output_directory=/ImageNet2012/tfrecord \
                              --train_directory=/ImageNet2012/train \
                              --validation_directory=/ImageNet2012/val
```



### Key Changes in `build_imagenet_data.py`

``` Python

def _convert_to_example(filename, UID, image_buffer, label, synset, human, bbox,
                        height, width):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    bbox: list of bounding boxes; each box is a list of integers
      specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
      the same label as the image label.
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  for b in bbox:
    assert len(b) == 4
    # pylint: disable=expression-not-assigned
    [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
    # pylint: enable=expression-not-assigned

  colorspace = b'RGB'
  channels = 3
  image_format = b'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(str.encode(synset)),
        'image/format': _bytes_feature(image_format),   
        'image/ID': _int64_feature(UID),
        'image/filename': _bytes_feature(os.path.basename(str.encode(filename))),
        'image/encoded': _bytes_feature(image_buffer)}))
  return example
  ```
  
  ```Python
  def _process_dataset(name, directory, num_shards, synset_to_human,
                     image_to_bboxes):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'
    image_to_bboxes: dictionary mapping image file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  """
  filenames, synsets, labels = _find_image_files(directory, FLAGS.labels_file)
  #print (len(filenames))
  
  file_id_dict = dict(zip(filenames,list(range(len(filenames))) ))  
  
  humans = _find_human_readable_labels(synsets, synset_to_human)
  bboxes = _find_image_bounding_boxes(filenames, image_to_bboxes)
  _process_image_files(name, filenames, file_id_dict, synsets, labels,
                       humans, bboxes, num_shards)
```


