# NER
Recognizing mountain entities in the text. Bert-based

## Setting up the repository
To set up this repository you'll need:

1) Run `pip install -r requirements.txt`
2) Create folder `weights` in the root of this repository.
3) Copy weights from the Google Drive to the folder `weights`

Also I have to say that tests showed, that the best models are `bert_v2-test` and `bert_v3-test`.

## Project Structure
To make project feasible for understanding, code and data were divided into different folders. Also data hase subfolders. Project has the following structure:
```
.
├── bert
├── data
│   ├── data_with_tags
│   │   ├── final
│   │   ├── natural
│   │   └── synthetic
│   ├── generated_raw_sentences
│   ├── logs
│   └── samples
└── misc
```
Let's look closer at main folders.

### Folder 'bert'
```
./bert/
├── __init__.py
├── dataset.py
├── model.py
├── tokenizer.py
├── utils.py
└── visualize.py
```

There are three main files:
1. *dataset.py* contains class BertDataset which is being used by the train.py script.
2. *model.py* main file where model is defined.
3. *tokenizer.py* contains class Tokenizer, as well as child of it – BertTokenizer, which is used in BertDataset for tokenization of sentences.
4. *utils.py* contains all function for evaluating metrics.
5. *visualize.py* is for visualizing metrics from raw data during training.

### Folder 'data'
```
./data/
├── data_with_tags
│   ├── final
│   ├── natural
│   └── synthetic
├── generated_raw_sentences
├── logs
└── samples
```

The main folder here is 'data_with_tags', which contains folders 'final', 'natural' and 'synthetic'. This folders are being used in dataset generation in 'data_generation.ipynb'. Folder 'final' is where the dataset is being stored after all preprocessing, and others represents discrete datasets from different sources.

1. Folder 'logs/openai' is where stored all info about recieved responses from OpenAI API.
2. Folder 'logs/train' is where all information about training model is being stored. There are also graphics of training.

## Training

For the training of this model you will need a dataset, consisting of columns 'sentence', 'tags'. For example:
<table>
    <tr>
        <th>sentence</th>
        <th>tags</th>
    </tr>
    <tr>
        <td>The Appalachian Mountains, stretching across the eastern United States, boast the ancient peaks of the Great Smoky Mountains, a haven for biodiversity and hiking trails like the Appalachian Trail.</td>
        <td>['O', 'B-mount', 'I-mount', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-mount', 'I-mount', 'I-mount', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-mount', 'I-mount', 'O']</td>
    </tr>
    <tr>
        <td>But this fire disaster happened over 110 years ago in the Northern Rocky Mountains of Idaho and Montana.</td>
        <td>['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-mount', 'I-mount', 'I-mount', 'O', 'O', 'O', 'O', 'O']</td>
    </tr>
    <tr>
        <td>...</td>
        <td>...</td>
    </tr>
</table>

As you can see only three classes are being used:
1. 'O' represents null class, means that this token is out of our interest.
2. 'B-mount' represents the beginning of our mountain entity.
3. 'I-mount' represents the continuation of out mountain entity.

To start training use the following train.py CLI:
```
options:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
                        Flag indicates number of epochs in training loop.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Sets batch-size.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Option for specifying learning rate in SGD.
  --dataset DATASET     Path to the dataset.
  --device DEVICE       Specify the device on which you will train model.
  --output_folder OUTPUT_FOLDER
                        Specify the name of a folder, which will contain final weights.
```

To start training model run the command:
```bash
$ python train.py -e 4 -b 4 -lr 0.004 --dataset data/data_with_tags/final/fine-tune-test-data-test-2.csv --output_folder bert_finetuned
```

After this dataset will be loaded and training will start. After completing training you will see folder 'bert_finetuned' in the root direcoty that contains weights for the model. Those weights will be used in inference.py script.

While training all logs are being stored in data/logs/train folder.

## Inferencing
To inference model on desirable text all you need is to run inference.py with the following CLI:
```
options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to the folder with information about model
  -i INPUT, --input INPUT
                        Path to the file with input.
  --output_format OUTPUT_FORMAT
                        Choose between JSON, HTML formats. Only JSON is supported right now.
  -o OUTPUT, --output OUTPUT
                        Name of the file where output will be stored.
  -d DEVICE, --device DEVICE
                        On which device to run the model.
```

The example of inferense providing you have file 'text.txt' with containing text:
```
python3 -m bert_finetuned -i text.txt
```

Model will generate json file, where mountain entity will be inside '[\BEGIN]' and '[\END]' tags.

# Testing
Currently the best model is `bert_v2-test` which resides on a Google Drive. As I mentioned before, it's better to put weights into a specifically created `weights` folder, but option `-m` in CLI of `inference.py` allows you to choose weights from anywhere else on your computer. 

For example, let's consider the following generated by ChatGPT 3.5 text (this file located in a folder `examples`):
```
I want to break free, Everest!
Nestled in the heart of the Rockies, Mount Celestia stands as a majestic sentinel, its snow-capped peak touching the sky.
The formidable Mount Embercore, with its rugged slopes and ancient glaciers, dominates the landscape of the northern wilderness.
Hidden among the clouds, the elusive Azure Peak boasts breathtaking vistas and is revered by mountaineers seeking the ultimate alpine challenge.
Some mountains were formed by the activity of volcanoes. 
Scientists believe that most volcanic mountains are made up of rock that melted deep within Earth. 
The rock rose through Earth's surface, or crust. It then flowed onto the surface in the form of lava. 
The lava, along with volcanic dust, built up to form mountains. 
Volcanic mountains are typically steep and cone shaped. 
Mount Fuji in Japan, Mount Kilimanjaro in Africa, and Mount Rainier in the United States are examples of volcanic mountains.
Other mountains were formed by movements within Earth's surface, or crust. 
The theory called plate tectonics explains this type of mountain building. 
Earth's surface is divided into huge pieces called plates, which move very slowly. 
The continents sit on top of the plates and move with them. 
At times the plates collide, forcing the rock upward. 
The Himalayas of Asia are an example of this type of mountain chain. 
They were formed when a plate carrying India collided with the Asian plate.
```

Run the command:

```
$ python inference.py -m weights/bert_v2-test -i examples/text.txt
```

The output will be saved in a file named `inference.json`, and will be as follows:

```json
{
  "sentences": [
    "I want to break free , Everest ! ",
    "Nestled in the heart of the Rockies , [\\BEGIN] Mount Celestia [\\END] stands as a majestic sentinel , its snow-capped peak touching the sky . ",
    "The formidable [\\BEGIN] Mount Embercore [\\END] , with its rugged slopes and ancient glaciers , dominates the landscape of the northern wilderness . ",
    "Hidden among the clouds , the elusive [\\BEGIN] Azure Peak [\\END] boasts breathtaking vistas and is revered by mountaineers seeking the ultimate alpine challenge . ",
    "Some mountains were formed by the activity of volcanoes . ",
    "Scientists believe that most volcanic mountains are made up of rock that melted deep within Earth . ",
    "The rock rose through Earth 's surface , or crust . ",
    "It then flowed onto the surface in the form of lava . ",
    "The lava , along with volcanic dust , built up to form mountains . ",
    "Volcanic mountains are typically steep and cone shaped . ",
    "[\\BEGIN] Mount Fuji [\\END] in Japan , [\\BEGIN] Mount [\\END] Kilimanjaro in Africa , and [\\BEGIN] Mount Rainier [\\END] in the United States are examples of volcanic mountains . ",
    "Other mountains were formed by movements within Earth 's surface , or crust . ",
    "The theory called plate tectonics explains this type of mountain building . ",
    "Earth 's surface is divided into huge pieces called plates , which move very slowly . ",
    "The continents sit on top of the plates and move with them . ",
    "At times the plates collide , forcing the rock upward . ",
    "The [\\BEGIN] Himalayas [\\END] of Asia are an example of this type of mountain chain . ",
    "They were formed when a plate carrying India collided with the Asian plate . "
  ]
}
```

Let's inference it on another texts called `examples/custom_text.txt`:

```
$ python inference.py -m weights/bert_v2-test -i examples/custom_text.txt
```

The result is:
```json
{
  "sentences": [
    "Mount Everest the great rock touching the sky . ",
    "So many have tried to climb it , but very few actually did it . ",
    "The origin of this mountain is graceful [\\BEGIN] Himalayas [\\END] locatead right in the middle of Central Asia . ",
    "Also there are [\\BEGIN] Southern Alps [\\END] . ",
    "Venturing into the realm of cosmic nomenclature , [\\BEGIN] Quasar Quinze [\\END] stands as a beacon in the night sky . ",
    "Named for its celestial associations , this mountain invites stargazers and astronomers to witness the wonders of the universe from its lofty heights . ",
    "The marriage of earth and sky takes on new meaning at [\\BEGIN] Quasar Quinze [\\END] . ",
    "While the name [\\BEGIN] Everest [\\END] might be familiar , add `` [\\BEGIN] Enchanted [\\END] '' to it , and a new realm of wonder opens up . ",
    "[\\BEGIN] Enchanted Everest [\\END] transcends the physical challenges of climbing the world 's tallest peak ; it hints at a spiritual journey , a transformation that occurs as one ascends to new heights . ",
    "It 's not merely a mountain ; it 's a magical ascent into the unknown . ",
    "A name that speaks of brightness and warmth , [\\BEGIN] Radiant Ridge [\\END] bathes in the golden glow of sunrise and sunset . ",
    "The ridge , with its undulating contours , seems to catch the first and last rays of sunlight , creating a spectacle that is both dazzling and serene . ",
    "[\\BEGIN] Radiant Ridge [\\END] is a canvas painted with nature 's most vibrant palette . "
  ]
}
```