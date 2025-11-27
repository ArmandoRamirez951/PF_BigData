<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Piedra, Papel o Tijera AI</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
    <style>
        body { font-family: 'Arial', sans-serif; text-align: center; background-color: #f0f0f0; }
        h1 { color: #333; }
        #game-container { position: relative; display: inline-block; margin-top: 20px; }
        video { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); transform: scaleX(-1); } /* Efecto espejo */
        #status { margin: 15px; font-size: 1.2em; font-weight: bold; color: #555; }
        button {
            padding: 15px 30px; font-size: 18px; cursor: pointer;
            background-color: #007bff; color: white; border: none; border-radius: 5px; margin: 10px;
        }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        #result-area { margin-top: 20px; font-size: 1.5em; padding: 20px; background: white; border-radius: 10px; display: none; }
        .win { color: green; }
        .lose { color: red; }
        .draw { color: orange; }
    </style>
</head>
<body>

    <h1>🎮 Piedra, Papel o Tijera IA</h1>
    
    <div id="game-container">
        <video id="webcam" width="448" height="448" autoplay playsinline></video>
    </div>

    <div id="status">Cargando modelo...</div>

    <div>
        <button id="btn-play" onclick="jugarRonda()" disabled>🖐 Iniciar Ronda</button>
        <button id="btn-score" onclick="verPuntaje()">🏆 Ver Puntaje</button>
    </div>

    <div id="result-area">
        <p id="prediction-text">Tú: ...</p>
        <p id="cpu-text">CPU: ...</p>
        <h2 id="final-result">RESULTADO</h2>
    </div>

    <script>
        // ---------------- CONFIGURACIÓN ----------------
        // Asegúrate de poner aquí las mismas etiquetas que tienes en labels.txt
        const LABELS = ["Piedra", "Papel", "Tijera", "Fondo", "Nada"]; 
        const LABELS_VALIDAS = ["Piedra", "Papel", "Tijera"];
        const MODEL_URL = './modelo_web/model.json'; // Ruta relativa a tu archivo convertido

        let model = null;
        let webcamElement = document.getElementById('webcam');
        let puntaje = { ganadas: 0, perdidas: 0 };
        let isPlaying = false;

        // ---------------- CARGAR MODELO Y CÁMARA ----------------
        async function init() {
            try {
                // Cargar modelo
                console.log("Cargando modelo...");
                // Nota: Usamos loadGraphModel porque convertimos un SavedModel
                model = await tf.loadGraphModel(MODEL_URL); 
                console.log("Modelo cargado.");

                // Configurar cámara
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                webcamElement.srcObject = stream;
                
                await new Promise(resolve => webcamElement.onloadedmetadata = resolve);
                document.getElementById('status').innerText = "Modelo listo. ¡Presiona Iniciar!";
                document.getElementById('btn-play').disabled = false;
            } catch (error) {
                console.error(error);
                document.getElementById('status').innerText = "Error: " + error.message;
                alert("Asegúrate de ejecutar esto en un servidor local (http://localhost), no abriendo el archivo directamente.");
            }
        }

        // ---------------- LÓGICA DEL JUEGO ----------------
        async function jugarRonda() {
            if (isPlaying) return;
            isPlaying = true;
            document.getElementById('btn-play').disabled = true;
            document.getElementById('result-area').style.display = 'none';
            
            let conteoPredicciones = [];
            let tiempoRestante = 3; // Segundos de "escaneo" (reducido de 5 a 3 para web)

            document.getElementById('status').innerText = "¡Prepara tu mano! Escaneando...";

            // Intervalo para capturar predicciones durante X segundos
            let intervalo = setInterval(async () => {
                const prediccion = await predecirImagen();
                if (LABELS_VALIDAS.includes(prediccion)) {
                    conteoPredicciones.push(prediccion);
                }
            }, 100); // Predecir cada 100ms

            // Finalizar ronda después del tiempo
            setTimeout(() => {
                clearInterval(intervalo);
                finalizarRonda(conteoPredicciones);
            }, tiempoRestante * 1000);
        }

        async function predecirImagen() {
            if (!model) return "Nada";

            // tf.tidy limpia la memoria de tensores intermedios automáticamente
            return tf.tidy(() => {
                // 1. Convertir imagen de webcam a tensor
                let img = tf.browser.fromPixels(webcamElement);
                
                // 2. Preprocesamiento (igual que en tu Python: resize 224, normalizar -1 a 1)
                img = tf.image.resizeBilinear(img, [224, 224]); // Resize
                img = img.expandDims(0); // Añadir batch dimension
                img = img.toFloat().div(127.5).sub(1.0); // Normalización

                // 3. Predicción
                const output = model.predict(img); // Puede devolver un objeto o tensor
                
                // Manejar si el output es un objeto (común en GraphModel) o un tensor directo
                const predictions = output.arraySync ? output.arraySync()[0] : output['Identity'].arraySync()[0]; 

                // 4. Encontrar el índice más alto
                let maxIndex = predictions.indexOf(Math.max(...predictions));
                return LABELS[maxIndex];
            });
        }

        function finalizarRonda(listaPredicciones) {
            isPlaying = false;
            document.getElementById('btn-play').disabled = false;

            // Encontrar el elemento más común (Moda)
            if (listaPredicciones.length === 0) {
                document.getElementById('status').innerText = "No detecté nada válido. Intenta de nuevo.";
                return;
            }

            // Lógica similar a Counter(list).most_common(1)
            const count = {};
            listaPredicciones.forEach(item => count[item] = (count[item] || 0) + 1);
            const jugador = Object.keys(count).reduce((a, b) => count[a] > count[b] ? a : b);

            // Turno CPU
            const computadora = LABELS_VALIDAS[Math.floor(Math.random() * LABELS_VALIDAS.length)];
            
            // Calcular ganador
            const resultado = obtenerResultado(jugador, computadora);

            // Mostrar resultados
            mostrarResultadoUI(jugador, computadora, resultado);
        }

        function obtenerResultado(jugador, computadora) {
            if (jugador === computadora) return "Empate";
            if ((jugador === "Piedra" && computadora === "Tijera") ||
                (jugador === "Papel" && computadora === "Piedra") ||
                (jugador === "Tijera" && computadora === "Papel")) {
                puntaje.ganadas++;
                return "Ganaste";
            } else {
                puntaje.perdidas++;
                return "Perdiste";
            }
        }

        function mostrarResultadoUI(jugador, cpu, resultado) {
            const resArea = document.getElementById('result-area');
            const finalRes = document.getElementById('final-result');
            
            resArea.style.display = 'block';
            document.getElementById('prediction-text').innerText = `Tú elegiste: ${jugador}`;
            document.getElementById('cpu-text').innerText = `CPU eligió: ${cpu}`;
            
            finalRes.innerText = resultado.toUpperCase();
            finalRes.className = ""; // Reset class
            
            if (resultado === "Ganaste") finalRes.classList.add("win");
            else if (resultado === "Perdiste") finalRes.classList.add("lose");
            else finalRes.classList.add("draw");

            document.getElementById('status').innerText = "Ronda terminada.";
        }

        function verPuntaje() {
            alert(`Partidas ganadas: ${puntaje.ganadas}\nPartidas perdidas: ${puntaje.perdidas}`);
        }

        // Iniciar al cargar la página
        init();

    </script>
</body>
</html>
