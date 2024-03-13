#  Funciones para el pre-procesamiento
import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model



Model_Path= 'modelos/'

def completar_array_modelo(modelo, array_test):
  #tamaño capa de entrada
  forma_entrada = modelo.input_shape[1:]

  #completar el array
  array_complete = np.array(array_test).reshape(1, -1)
  array_complete = np.pad(array_complete, ((0, 0), (0, forma_entrada[0] - array_complete.shape[1])), 'constant', constant_values=0)
  return array_complete

#Variables de tiempo propias de cada el ejercicio
def tiempo(ejercicio):
  tiempos = {
    'sentadilla': {'rom': 2.9, 'vmed': 1.5, 'vmax':2.8},
    'pressbanca': {'rom': 0.4, 'vmed': 1.6, 'vmax': 2.2},
    'flexiones': {'rom': 2.9, 'vmed': 1.5, 'vmax': 2.8},
    'muerto': {'rom': 2.9, 'vmed': 1.5, 'vmax': 2.8}
  }
  if ejercicio in tiempos:
      return tiempos[ejercicio]['rom'], tiempos[ejercicio]['vmed'], tiempos[ejercicio]['vmax']
  else:
      print(f"Error: el ejercicio '{ejercicio} no está definido en la función")
      return None
def sqe_input_pred(input_values, ejercicio):
    target_columns = ['rom', 'vmed', 'vmax']  # Columnas de salida esperadas

    # Obtener los tiempos correspondientes al ejercicio
    tiempo_rom, tiempo_Vmed, tiempo_Vmax = tiempo(ejercicio)

    # Longitud de la secuencia según el modelo
    len_rom = round(tiempo_rom / (1 / 50))
    len_vmed = round(tiempo_Vmed / (1 / 50))
    len_vmax = round(tiempo_Vmax / (1 / 50))


    # Asegurarse de que la entrada tenga al menos la longitud máxima de las secuencias
    max_len = max(len_rom, len_vmed, len_vmax)
    if len(input_values[0]) < max_len:
      print("Error: La longitud de datos de entrada es menor que la longitud de la secuencia requerida.")
      return None, None, None

    last_samples = input_values[0]


    # Tomar las últimas muestras según la longitud de cada secuencia
    last_samples_rom = last_samples[-len_rom*3:]
    last_samples_vmed = last_samples[-len_vmed*3:]
    last_samples_vmax = last_samples[-len_vmax*3:]

    # Crear las secuencias de entrada para cada tiempo
    x_rom = np.array(last_samples_rom)

    x_vmed = np.array(last_samples_vmed)

    x_vmax = np.array(last_samples_vmax)



    # Normalizar los datos para cada secuencia
    x_rom  = normalize(x_rom.reshape(1,-1)).astype('float32')
    x_vmed = normalize(x_vmed.reshape(1,-1)).astype('float32')
    x_vmax = normalize(x_vmax.reshape(1,-1)).astype('float32')

    return x_rom, x_vmed, x_vmax



def Regression_models(ejercicio): #cargar modelos de regresión para ej:sentadilla
  if ejercicio == "sentadilla":

    model_rom = tf.keras.models.load_model(Model_Path + 'Model_Sentadillas_ROM_P3.h5')

    model_vmed = tf.keras.models.load_model(Model_Path + 'model_Sentadillas_VMED_P1.h5')

    model_vmax = tf.keras.models.load_model(Model_Path + 'Model_Sentadillas_VMAX_P1.h5')

    return model_rom, model_vmed, model_vmax

  elif ejercicio == "pressbanca":

    model_rom = tf.keras.models.load_model(Model_Path + 'Model_Pressbanca_ROM_P4.h5')

    model_vmed = tf.keras.models.load_model(Model_Path + 'Model_Pressbanca_Vmed_P6.h5')

    model_vmax = tf.keras.models.load_model(Model_Path + 'Model_Pressbanca_Vmax_P2.h5')
    return model_rom, model_vmed, model_vmax

  elif ejercicio == "flexiones":

    pass

  #     model_rom  = tf.keras.models.load_model(Model_Path+'')
  #     model_vmed = tf.keras.models.load_model(Model_Path+'')
  #     model_vmax = tf.keras.modelsload_model(Model_Path+'')
  else:

    return None
  #     model_rom  = tf.keras.models.load_model(Model_Path+'')
  #     model_vmed = tf.keras.models.load_model(Model_Path+'')
  #     model_vmax = tf.keras.models.load_model(Model_Path+'')




def predict_(model_1, model_2, model_3, x_1, x_2, x_3):

    pred_rom  = model_1.predict(x_1)
    pred_vmed = model_2.predict(x_2)
    pred_vmax = model_3.predict(x_3)

    return pred_rom, pred_vmed, pred_vmax

