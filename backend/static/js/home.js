// frontend/static/js/home.js


const letraSorteadaElement = document.getElementById('letra-sorteada');
const imagemLibrasElement = document.getElementById('imagem-libras');
const botaoNovaLetra = document.getElementById('botao-nova-letra');
const letras = 'abcdefghijklmnopqrstuvwxyz';
const caminhoImagens = '/static/imagens/'; // Ajuste o caminho se sua pasta 'imagens' estiver em outro lugar dentro de 'static'


function gerarLetraAleatoria() {
 const indiceAleatorio = Math.floor(Math.random() * letras.length);
 return letras.charAt(indiceAleatorio);
}


function atualizarLetraEImagem() {
 const letra = gerarLetraAleatoria();
 letraSorteadaElement.textContent = letra.toUpperCase();
 imagemLibrasElement.src = caminhoImagens + letra.toLowerCase() + '.jpeg';
 imagemLibrasElement.alt = `Imagem da letra ${letra} em Libras`;
}


// Gerar a primeira letra ao carregar a página
atualizarLetraEImagem();


// Adicionar evento de clique ao botão
botaoNovaLetra.addEventListener('click', atualizarLetraEImagem);