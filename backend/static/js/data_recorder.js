const webcamLive = document.getElementById('webcam-live');
const letterSelect = document.getElementById('letter-select');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusMessage = document.getElementById('status-message');
let recordingInterval;
let imageCount = 0;

// Função para iniciar a webcam de forma mais robusta
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamLive.srcObject = stream;
        
        // Retorna uma Promise que resolve quando o vídeo está pronto para rodar
        await new Promise((resolve) => {
            webcamLive.onloadedmetadata = () => {
                resolve();
            };
        });

        webcamLive.play(); // Garante que o vídeo está rodando
        
        console.log(`Webcam iniciada com sucesso. Dimensões: ${webcamLive.videoWidth}x${webcamLive.videoHeight}`);

    } catch (err) {
        statusMessage.textContent = 'Erro ao acessar a câmera. Verifique se ela está conectada e se você concedeu a permissão no navegador.';
        console.error("Erro ao acessar a câmera: ", err);
    }
}


async function startRecording() {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    const selectedLetter = letterSelect.value;
    statusMessage.textContent = `Gravando imagens para a letra ${selectedLetter}...`;

    recordingInterval = setInterval(async () => {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = webcamLive.videoWidth;
        tempCanvas.height = webcamLive.videoHeight;
        const tempContext = tempCanvas.getContext('2d');
        tempContext.drawImage(webcamLive, 0, 0, tempCanvas.width, tempCanvas.height);
        
        const imageDataUrl = tempCanvas.toDataURL('image/jpeg');

        try {
            const response = await fetch('/record_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageDataUrl, label: selectedLetter })
            });

            const result = await response.json();
            if (result.success) {
                imageCount++;
                statusMessage.textContent = `Gravando... ${imageCount} imagens salvas para a letra ${selectedLetter}.`;
            } else {
                statusMessage.textContent = `Erro ao salvar imagem: ${result.message}`;
            }
        } catch (err) {
            statusMessage.textContent = 'Erro de conexão com o servidor.';
            console.error("Erro ao enviar imagem: ", err);
        }
    }, 1000); // Tira uma foto a cada 1 segundo
}

function stopRecording() {
    clearInterval(recordingInterval);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusMessage.textContent = `Gravação parada. Total de ${imageCount} imagens salvas.`;
    imageCount = 0;
}

startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// Inicia a câmera ao carregar a página
window.onload = startWebcam;