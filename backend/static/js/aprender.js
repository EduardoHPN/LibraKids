// backend/static/js/aprender.js

const webcamPreview = document.getElementById('webcam-preview');
const overlayCanvas = document.getElementById('overlay-canvas');
const letraDesafioElement = document.getElementById('letra-desafio');
const imagemReferenciaElement = document.getElementById('imagem-referencia');
const feedbackMessage = document.getElementById('feedback-message');
const contexto = overlayCanvas.getContext('2d');
const letras = 'ABCDEFGILMNOPQRSTUVXY';
const caminhoImagens = '/static/imagens/';

// Função para iniciar a webcam e o canvas
async function iniciarWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamPreview.srcObject = stream;
        webcamPreview.play();

        // Espera o vídeo carregar para definir o tamanho do canvas
        webcamPreview.addEventListener('loadeddata', () => {
            overlayCanvas.width = webcamPreview.videoWidth;
            overlayCanvas.height = webcamPreview.videoHeight;
        });

    } catch (err) {
        console.error("Erro ao acessar a câmera: ", err);
        alert("Não foi possível acessar a câmera. Verifique as permissões do navegador.");
    }
}

// Função para gerar uma nova letra aleatória
function gerarNovaLetra() {
    const letraAleatoria = letras[Math.floor(Math.random() * letras.length)];
    letraDesafioElement.textContent = letraAleatoria;
    imagemReferenciaElement.src = caminhoImagens + letraAleatoria.toLowerCase() + '.jpg';
    imagemReferenciaElement.alt = `Imagem da letra ${letraAleatoria} em Libras`;
    feedbackMessage.textContent = '';
    contexto.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height); // Limpa o canvas
}

// Função para capturar e enviar a imagem para o backend
async function capturarEVerificar() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = webcamPreview.videoWidth;
    tempCanvas.height = webcamPreview.videoHeight;
    const tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(webcamPreview, 0, 0, tempCanvas.width, tempCanvas.height);
    
    // Converte a imagem do canvas para Base64
    const imageDataUrl = tempCanvas.toDataURL('image/jpg');

    const targetLetter = letraDesafioElement.textContent;

    try {
        const response = await fetch('/predict_letter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageDataUrl, target_letter: targetLetter })
        });

        const result = await response.json();
        
        if (result.success) {
            // Limpa o canvas e desenha a caixa de detecção da mão
            contexto.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            if (result.box) {
                const [x_min, y_min, x_max, y_max] = result.box;
                contexto.beginPath();
                contexto.rect(x_min, y_min, x_max - x_min, y_max - y_min);
                contexto.lineWidth = 4;
                contexto.strokeStyle = result.correct ? 'green' : 'red';
                contexto.stroke();
            }

            if (result.correct) {
                feedbackMessage.textContent = `Correto! É a letra ${result.predicted_letter}.`;
                feedbackMessage.style.color = 'green';
                // Espera um pouco e gera a próxima letra
                setTimeout(gerarNovaLetra, 2000);
            } else {
                if (result.predicted_letter === 'Nenhuma mão detectada') {
                    feedbackMessage.textContent = `Tente fazer a letra mais claramente.`;
                } else {
                    feedbackMessage.textContent = `Incorreto. Você fez a letra ${result.predicted_letter}. Tente novamente!`;
                }
                feedbackMessage.style.color = 'red';
            }
        } else {
            feedbackMessage.textContent = 'Erro ao processar a imagem. Tente novamente.';
            feedbackMessage.style.color = 'gray';
            console.error(result.message);
        }

    } catch (err) {
        feedbackMessage.textContent = 'Ocorreu um erro de conexão.';
        feedbackMessage.style.color = 'gray';
        console.error("Erro na comunicação com o backend: ", err);
    }
}

// Inicia a webcam e gera a primeira letra quando a página carrega
window.onload = function() {
    iniciarWebcam();
    gerarNovaLetra();
    // Chama a função de verificação a cada 2 segundos
    setInterval(capturarEVerificar, 2000);
};