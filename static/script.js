document.getElementById('chat-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const chatInput = document.getElementById('user-input');
    const message = chatInput.value;
    chatInput.value = '';

    const chatLog = document.getElementById('messages');
    const userMessage = document.createElement('div');
    userMessage.classList.add('user-message');
    userMessage.textContent = `You: ${message}`;
    chatLog.appendChild(userMessage);

    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: message })
    });

    const data = await response.json();
    const botMessage = document.createElement('div');
    botMessage.classList.add('bot-message');
    botMessage.textContent = `Bot: ${data.answer}`;
    chatLog.appendChild(botMessage);

    chatLog.scrollTop = chatLog.scrollHeight;
});
