class DocumentQA {
    constructor() {
        this.API_URL = 'http://localhost:8000';
        this.messageContainer = document.getElementById('chatMessages');
        this.questionInput = document.getElementById('questionInput');
        this.sendButton = document.getElementById('sendButton');
        this.statusIndicator = document.getElementById('statusIndicator');
        
        this.initialize();
    }

    initialize() {
        this.addEventListeners();
        this.checkSystemStatus();
    }

    addEventListeners() {
        this.questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleQuestion();
            }
        });

        this.sendButton.addEventListener('click', () => {
            this.handleQuestion();
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch(`${this.API_URL}/status`);
            const data = await response.json();
            
            if (data.system_ready) {
                this.statusIndicator.textContent = `System ready `;
                this.statusIndicator.style.color = '#28a745';
            } else {
                this.statusIndicator.textContent = 'System initializing...';
                this.statusIndicator.style.color = '#ffc107';
            }
        } catch (error) {
            this.statusIndicator.textContent = 'System unavailable';
            this.statusIndicator.style.color = '#dc3545';
        }
    }

    async handleQuestion() {
        const question = this.questionInput.value.trim();
        if (!question) return;

        this.setInputState(true);
        this.addMessage('user', question);
        this.questionInput.value = '';

        try {
            const response = await this.sendQuestion(question);
            this.addMessage('assistant', response.response);
        } catch (error) {
            this.addMessage('assistant', 'Error: Unable to process your question.');
        } finally {
            this.setInputState(false);
        }
    }

    async sendQuestion(question) {
        const response = await fetch(`${this.API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: question }),
        });

        if (!response.ok) {
            throw new Error('Failed to get response');
        }

        return await response.json();
    }

    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        messageDiv.textContent = content;
        this.messageContainer.appendChild(messageDiv);
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }

    setInputState(disabled) {
        this.questionInput.disabled = disabled;
        this.sendButton.disabled = disabled;
        this.sendButton.textContent = disabled ? 'Processing...' : 'Send';
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DocumentQA();
});
