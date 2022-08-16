import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier

def getBaseModel(IMG_SHAPE):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    return base_model

def getBaseModel_KNN(IMG_SHAPE):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    base_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    return base_model

def train_KNN(activations, y):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(activations, y)

    return classifier


def saveClassandLabel(classes, y):

    f = open("saved_model/labels.txt", "w", encoding="cp949")
    for c in classes:
        f.write(c + "\n")
    f.close()

    f = open("saved_model/ylist.txt", "w", encoding="cp949")
    for c in y:
        f.write(str(c) + "\n")
    f.close()

    print("\nyour model has been saved\n")

    

def saveClassandLabel_object(classes, y):

    f = open("saved_model/labels_object.txt", "w", encoding="cp949")
    for c in classes:
        f.write(c + "\n")
    f.close()

    f = open("saved_model/ylist_object.txt", "w", encoding="cp949")
    for c in y:
        f.write(str(c) + "\n")
    f.close()

    print("\nyour model has been saved\n")

if __name__ == "__main__":
    classes = ["a", "b"]
    saveClassandLabel(classes)
