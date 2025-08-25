// backend/static/js/menu.js

document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.querySelector('.sidebar');
    const toggleButton = document.getElementById('menu-toggle');
    const contentWrapper = document.querySelector('.content-wrapper');

    toggleButton.addEventListener('click', () => {
        sidebar.classList.toggle('is-open');
    });
});