from flask import Flask, request, jsonify
from utils_api import predict_, Regression_models, tiempo, sqe_input_pred




app = Flask(__name__)

#Parametrosd de entrada comunes:
parametros_ejercicio = {
    "sentadilla": {
        "campos_entrada": ["edad", "sexo", "altura", "peso", "acc", "ejercicio"]},
    "pressbanca": {
        "campos_entrada": ["edad", "sexo", "altura", "peso", "acc", "ejercicio"]},
}

@app.route('/modelosRegresion/ejercicios', methods=['GET'])
def get_available_exercises():
    exercises = ["sentadilla", "pressbanca"] #, "flexiones", "peso muerto"]
    return jsonify({"ejercicios disponibles": exercises}), 200

@app.route('/modelosRegresion/parametros', methods=['POST'])
def get_model_parameters():
    campos_entrada = request.get_json()


    if not campos_entrada or 'ejercicio' not in campos_entrada:
        return jsonify({"error": "Parámetros de entrada incorrectos"}), 400

    ejercicio_solicitado = campos_entrada['ejercicio']

    if ejercicio_solicitado not in parametros_ejercicio:
        return jsonify({"error": "Ejercicio no encontrado"}), 404

    parametros = parametros_ejercicio[ejercicio_solicitado]

    parameters = {
        "campos de entrada": parametros["campos_entrada"],
        "Tiempos ROM, Vmed, Vmax": tiempo(ejercicio_solicitado)}

    return jsonify({"campos entrada": parameters}), 200

@app.route('/modelosRegresion/prediccion', methods=['POST'])
def make_prediction():
    data = request.get_json()
    ejercicio = data.get('ejercicio')
    edad = data.get('edad')
    altura = data.get('altura')
    sexo = data.get('sexo')
    peso = data.get('peso')
    acc_data = data.get('accData')
    gir_data = data.get('girData')

    model_rom, model_vmed, model_vmax = Regression_models(ejercicio)

    if model_rom is None or model_vmed is None or model_vmax is None:
        return jsonify({"error": "Ejercicio no encontrado"}), 404

    # Procesa los datos de entrada
    # Realiza predicciones utilizando los modelos
    X = [#age, # height, # gender,# weight,
               acc_data ]#,gir_data ]

    x_rom, x_vmed, x_vmax = sqe_input_pred(X, ejercicio)

    if x_rom is None or x_vmed is None or x_vmax is None:
        return jsonify({"error": "Longitud de datos insuficiente"}), 404

    y_1, y2, y3 = predict_(model_rom, model_vmed, model_vmax, x_rom, x_vmed, x_vmax)

    # Resultados
    results = {
        "rom_prediction": y_1.tolist(),
        "vmed_prediction": y2.tolist(),
        "vmax_prediction": y3.tolist()
    }

    return jsonify(results), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
