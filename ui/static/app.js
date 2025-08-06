// Global error handler
window.addEventListener('error', function(e) {
    console.error('Global error caught:', e.error);
    console.error('Error details:', {
        message: e.message,
        filename: e.filename,
        lineno: e.lineno,
        colno: e.colno
    });
    
    // Try to show error message if possible
    try {
        addErrorMessage('An unexpected error occurred. Please refresh the page.');
    } catch (err) {
        console.error('Failed to show error message:', err);
        alert('An unexpected error occurred. Please refresh the page.');
    }
});

// Document RAG Chat Interface JavaScript

let currentSessionId = null;
let conversationHistory = [];
let isProcessing = false;
let isInitialized = false; // Global initialization flag

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing application...');
    
    // Wait a bit to ensure all elements are ready
    setTimeout(() => {
        initializeApplication();
    }, 100);
});

function initializeApplication() {
    console.log('Initializing application...');
    
    // Check if all required elements exist
    const requiredElements = [
        'queryInput',
        'submitBtn', 
        'submitText',
        'loadingSpinner',
        'citationFormat',
        'includeContext',
        'charCount',
        'conversationHistory'
    ];
    
    const missingElements = [];
    for (const elementId of requiredElements) {
        const element = document.getElementById(elementId);
        if (!element) {
            missingElements.push(elementId);
        }
    }
    
    if (missingElements.length > 0) {
        console.error('Missing required elements:', missingElements);
        addErrorMessage('UI initialization failed. Please refresh the page.');
        return;
    }
    
    console.log('All required elements found, proceeding with initialization...');
    
    // Generate a session ID
    currentSessionId = generateSessionId();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize UI state
    updateCharacterCount();
    updateSubmitButton();
    
    // Focus on the query input
    const queryInput = document.getElementById('queryInput');
    if (queryInput) {
        queryInput.focus();
    }
    
    // Mark as initialized
    isInitialized = true;
    
    console.log('Application initialized successfully');
}

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    const queryInput = document.getElementById('queryInput');
    const submitBtn = document.getElementById('submitBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');
    
    if (!queryInput || !submitBtn) {
        console.error('Required elements for event listeners not found');
        return;
    }
    
    // Submit button click
    submitBtn.addEventListener('click', function(e) {
        e.preventDefault();
        submitQuery();
    });
    
    // Clear chat button click
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', function(e) {
            e.preventDefault();
            clearConversation();
        });
    }
    
    // Enter key in textarea (Enter to submit, Shift+Enter for new line)
    queryInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isProcessing && queryInput.value && queryInput.value.trim()) {
                submitQuery();
            }
        }
    });
    
    // Auto-resize textarea and update character count
    queryInput.addEventListener('input', function() {
        updateCharacterCount();
        autoResizeTextarea();
        updateSubmitButton();
    });
    
    // Focus management
    queryInput.addEventListener('focus', function() {
        this.parentElement.classList.add('focused');
    });
    
    queryInput.addEventListener('blur', function() {
        this.parentElement.classList.remove('focused');
    });
    
    console.log('Event listeners set up successfully');
}

function updateCharacterCount() {
    if (!isInitialized) return;
    
    const queryInput = document.getElementById('queryInput');
    const charCount = document.getElementById('charCount');
    
    if (!queryInput || !charCount) {
        console.error('Character count elements not found');
        return;
    }
    
    // Safely get input value with null check
    const inputValue = queryInput.value || '';
    const currentLength = inputValue.length;
    const maxLength = 2000;
    
    charCount.textContent = `${currentLength}/${maxLength}`;
    
    if (currentLength > maxLength * 0.9) {
        charCount.classList.add('warning');
    } else {
        charCount.classList.remove('warning');
    }
}

function autoResizeTextarea() {
    if (!isInitialized) return;
    
    const queryInput = document.getElementById('queryInput');
    if (!queryInput) {
        console.error('Query input element not found');
        return;
    }
    
    // Safely access scrollHeight with null check
    const scrollHeight = queryInput.scrollHeight || 0;
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(scrollHeight, 200) + 'px';
}

function updateSubmitButton() {
    if (!isInitialized) return;
    
    const queryInput = document.getElementById('queryInput');
    const submitBtn = document.getElementById('submitBtn');
    
    if (!queryInput || !submitBtn) {
        console.error('Submit button elements not found');
        return;
    }
    
    // Safely get input value with null check
    const inputValue = queryInput.value || '';
    const hasContent = inputValue.trim().length > 0;
    
    submitBtn.disabled = isProcessing || !hasContent;
}

async function submitQuery() {
    // Check if application is initialized
    if (!isInitialized) {
        console.error('Application not yet initialized');
        return;
    }
    
    if (isProcessing) return;
    
    const queryInput = document.getElementById('queryInput');
    const submitBtn = document.getElementById('submitBtn');
    const submitText = document.getElementById('submitText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const citationFormat = document.getElementById('citationFormat');
    const includeContext = document.getElementById('includeContext');
    
    // Check if all required elements exist
    if (!queryInput || !submitBtn || !submitText || !loadingSpinner || !citationFormat || !includeContext) {
        console.error('Required DOM elements not found:', {
            queryInput: !!queryInput,
            submitBtn: !!submitBtn,
            submitText: !!submitText,
            loadingSpinner: !!loadingSpinner,
            citationFormat: !!citationFormat,
            includeContext: !!includeContext
        });
        addErrorMessage('UI elements not found. Please refresh the page.');
        return;
    }
    
    // Safely get the query value with null check
    const query = queryInput.value ? queryInput.value.trim() : '';
    
    if (!query) {
        console.log('Empty query, not submitting');
        return;
    }
    
    // Set processing state
    isProcessing = true;
    submitBtn.disabled = true;
    submitText.style.display = 'none';
    loadingSpinner.style.display = 'inline';
    
    // Add user message to conversation
    console.log('About to add user message:', query);
    addUserMessage(query);
    console.log('User message added');
    
    // Clear input
    queryInput.value = '';
    updateCharacterCount();
    autoResizeTextarea();
    updateSubmitButton();
    
    // Add loading message
    const loadingMessageId = addLoadingMessage();
    
    try {
        // Prepare request data with safe value access
        const requestData = {
            query: query,
            session_id: currentSessionId,
            user_id: null,
            citation_format: citationFormat.value || 'apa',
            include_context: includeContext.checked !== undefined ? includeContext.checked : true
        };
        
        console.log('Sending request:', requestData);
        
        // Make API call
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Received response:', result);
        
        // Validate the response structure
        if (!result || typeof result !== 'object') {
            throw new Error('Invalid response format from server');
        }
        
        // Remove loading message and add assistant response
        removeLoadingMessage(loadingMessageId);
        addAssistantMessage(result);
        
    } catch (error) {
        console.error('Error:', error);
        removeLoadingMessage(loadingMessageId);
        
        // Safely get error message
        let errorMessage = 'An error occurred while processing your query';
        if (error && typeof error === 'object' && error.message) {
            errorMessage = error.message;
        } else if (typeof error === 'string') {
            errorMessage = error;
        }
        
        addErrorMessage(errorMessage);
    } finally {
        // Reset processing state with null checks
        isProcessing = false;
        
        if (submitBtn) {
            submitBtn.disabled = false;
        }
        
        if (submitText) {
            submitText.style.display = 'inline';
        }
        
        if (loadingSpinner) {
            loadingSpinner.style.display = 'none';
        }
        
        updateSubmitButton();
        
        // Focus back on input with null check
        if (queryInput) {
            queryInput.focus();
        }
    }
}

function addMessageToConversation(type, content, metadata = {}) {
    console.log('addMessageToConversation called:', { type, content, metadata });
    
    // Ensure content is a string
    let safeContent = content || '';
    if (typeof safeContent !== 'string') {
        console.warn('Content is not a string, converting:', safeContent);
        safeContent = String(safeContent);
    }
    
    const message = {
        id: 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9),
        type: type,
        content: safeContent,
        timestamp: new Date(),
        metadata: metadata || {}
    };
    
    console.log('Created message object:', message);
    conversationHistory.push(message);
    console.log('Message added to conversationHistory array. Current length:', conversationHistory.length);
    return message.id;
}

function addUserMessage(content) {
    console.log('addUserMessage called with:', content);
    const messageId = addMessageToConversation('user', content);
    console.log('Message ID generated:', messageId);
    displayUserMessage(content, messageId);
    return messageId;
}

function addAssistantMessage(result) {
    console.log('addAssistantMessage called with result:', result);
    
    // Handle both 'response' and 'answer' fields from the API
    const responseText = result.response || result.answer || 'No response received';
    
    const messageId = addMessageToConversation('assistant', responseText, {
        sources: result.sources || [],
        confidence: result.confidence || 0.8,
        processing_time: result.processing_time || 0,
        query_type: result.query_type || 'unknown'
    });
    
    console.log('Assistant message ID:', messageId);
    displayAssistantMessage(result, messageId);
    console.log('Assistant message displayed');
    return messageId;
}

function addErrorMessage(errorMessage) {
    console.log('Adding error message:', errorMessage);
    
    // Check if application is initialized
    if (!isInitialized) {
        console.error('Cannot add error message - application not initialized');
        alert('Error: ' + errorMessage);
        return null;
    }
    
    // Check if conversation history is ready
    const conversationHistory = document.getElementById('conversationHistory');
    if (!conversationHistory) {
        console.error('Cannot add error message - conversation history not ready');
        // Try to show error in a different way
        alert('Error: ' + errorMessage);
        return null;
    }
    
    const messageId = addMessageToConversation('error', errorMessage);
    displayErrorMessage(errorMessage, messageId);
    return messageId;
}

function displayErrorMessage(errorMessage, messageId) {
    const conversationHistory = document.getElementById('conversationHistory');
    
    if (!conversationHistory) {
        console.error('Conversation history element not found');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message error-message';
    messageDiv.id = messageId;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="error-content">
                <span class="error-icon">❌</span>
                <span class="error-text">Error: ${escapeHtml(errorMessage)}</span>
            </div>
            <div class="error-help">Please try again or contact support if the problem persists.</div>
        </div>
        <div class="message-timestamp">${formatTimestamp(new Date())}</div>
    `;
    
    conversationHistory.appendChild(messageDiv);
    scrollToBottom();
}

function addLoadingMessage() {
    const messageId = 'loading_' + Date.now();
    displayLoadingMessage(messageId);
    return messageId;
}

function displayUserMessage(content, messageId) {
    console.log('displayUserMessage called with:', content, messageId);
    const conversationHistory = document.getElementById('conversationHistory');
    console.log('Conversation history element:', conversationHistory);
    
    if (!conversationHistory) {
        console.error('Conversation history element not found');
        return;
    }
    
    // Ensure content is safe
    const safeContent = content || '';
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.id = messageId;
    
    // Remove debug styling and make it look professional
    messageDiv.innerHTML = `
        <div class="message-content">
            <p><strong>You asked:</strong> ${escapeHtml(safeContent)}</p>
        </div>
        <div class="message-timestamp">${formatTimestamp(new Date())}</div>
    `;
    
    console.log('Created message div:', messageDiv);
    conversationHistory.appendChild(messageDiv);
    console.log('Message appended to conversation history');
    console.log('Total messages in conversation history:', conversationHistory.children.length);
    
    // Scroll to bottom to show the new message
    scrollToBottom();
}

function displayAssistantMessage(result, messageId) {
    const conversationHistory = document.getElementById('conversationHistory');
    
    if (!conversationHistory) {
        console.error('Conversation history element not found');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.id = messageId;
    
    // Build sources HTML if available
    let sourcesHtml = '';
    if (result.sources && result.sources.length > 0) {
        // Deduplicate sources by URL
        const uniqueSources = deduplicateSources(result.sources);
        
        sourcesHtml = `
            <div class="message-sources">
                <details>
                    <summary>📚 Sources (${uniqueSources.length})</summary>
                    <div class="sources-list">
                        ${uniqueSources.map((source, index) => {
                            // Check if we have file link information
                            const hasGoogleDriveUrl = source.google_drive_url && source.google_drive_url.trim() !== '';
                            const hasSourceUrl = source.source_url && source.source_url.trim() !== '';
                            const hasUrl = source.url && source.url.trim() !== '';
                            
                            // Determine the best URL to display
                            let displayUrl = '';
                            let urlType = '';
                            
                            if (hasGoogleDriveUrl) {
                                displayUrl = source.google_drive_url;
                                urlType = 'Google Drive';
                            } else if (hasSourceUrl) {
                                displayUrl = source.source_url;
                                urlType = 'Source';
                            } else if (hasUrl) {
                                displayUrl = source.url;
                                urlType = 'Link';
                            }
                            
                            return `
                                <div class="source-item">
                                    <div class="source-title">
                                        ${source.title || `Document ${index + 1}`}
                                        ${displayUrl ? `
                                            <div class="file-link-indicator">
                                                <a href="${displayUrl}" target="_blank" title="Open ${urlType} link">
                                                    🔗 ${urlType} Link
                                                </a>
                                            </div>
                                        ` : ''}
                                    </div>
                                    <div class="source-content">
                                        ${source.content || 'Content preview not available'}
                                    </div>
                                    <div class="source-meta">
                                        <span>Score: ${Math.round((source.confidence || 0.8) * 100)}%</span>
                                        ${source.page_number ? `<span>Page: ${source.page_number}</span>` : ''}
                                        ${source.section_title ? `<span>Section: ${source.section_title}</span>` : ''}
                                        ${source.duplicateCount > 1 ? `<span>Found in ${source.duplicateCount} sections</span>` : ''}
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </details>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="response-text">
                ${formatResponseText(result.response || result.answer || 'No response received')}
            </div>
            ${sourcesHtml}
            <div class="message-meta">
                <span class="confidence">Confidence: ${Math.round((result.confidence || 0.8) * 100)}%</span>
                <span class="processing-time">Processed in ${(result.processing_time || 0).toFixed(2)}s</span>
                ${result.query_type ? `<span class="query-type">Query Type: ${result.query_type}</span>` : ''}
            </div>
        </div>
        <div class="message-timestamp">${formatTimestamp(new Date())}</div>
    `;
    
    conversationHistory.appendChild(messageDiv);
    scrollToBottom();
}

function displayLoadingMessage(messageId) {
    const conversationHistory = document.getElementById('conversationHistory');
    
    if (!conversationHistory) {
        console.error('Conversation history element not found');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message loading-message';
    messageDiv.id = messageId;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading-indicator">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span>Thinking...</span>
            </div>
        </div>
    `;
    
    conversationHistory.appendChild(messageDiv);
    scrollToBottom();
}

function removeLoadingMessage(messageId) {
    console.log('removeLoadingMessage called with ID:', messageId);
    
    // Validate messageId
    if (!messageId || typeof messageId !== 'string') {
        console.warn('Invalid messageId provided to removeLoadingMessage:', messageId);
        return;
    }
    
    const loadingMessage = document.getElementById(messageId);
    console.log('Loading message element found:', loadingMessage);
    
    if (loadingMessage) {
        console.log('Removing loading message');
        // Only remove if it's actually a loading message, not a user message
        if (loadingMessage.classList && loadingMessage.classList.contains('loading-message')) {
            loadingMessage.remove();
        } else {
            console.log('Not removing - not a loading message');
        }
    } else {
        console.log('Loading message not found');
    }
}

function formatResponseText(text) {
    // Handle null, undefined, or non-string values
    if (text == null) {
        return '';
    }
    
    const safeText = String(text);
    
    // Convert markdown-like formatting to HTML
    return escapeHtml(safeText)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
    // Handle null, undefined, or non-string values
    if (text == null) {
        return '';
    }
    
    const safeText = String(text);
    const div = document.createElement('div');
    div.textContent = safeText;
    return div.innerHTML;
}

function formatTimestamp(date) {
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) { // Less than 1 minute
        return 'Just now';
    } else if (diff < 3600000) { // Less than 1 hour
        const minutes = Math.floor(diff / 60000);
        return `${minutes}m ago`;
    } else if (diff < 86400000) { // Less than 1 day
        const hours = Math.floor(diff / 3600000);
        return `${hours}h ago`;
    } else {
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }
}

function scrollToBottom() {
    const conversationHistory = document.getElementById('conversationHistory');
    if (conversationHistory) {
        conversationHistory.scrollTop = conversationHistory.scrollHeight;
    }
}

function clearConversation() {
    if (!isInitialized) return;
    
    const conversationHistory = document.getElementById('conversationHistory');
    if (!conversationHistory) {
        console.error('Conversation history element not found');
        return;
    }
    
    // Keep only the welcome message
    const welcomeMessage = conversationHistory.querySelector('.assistant-message');
    conversationHistory.innerHTML = '';
    if (welcomeMessage) {
        conversationHistory.appendChild(welcomeMessage);
    }
    
    // Clear the conversation history array
    conversationHistory.length = 0;
    
    // Generate a new session ID
    currentSessionId = generateSessionId();
    
    console.log('Conversation cleared');
}

// Deduplicate sources based on their URLs
function deduplicateSources(sources) {
    const urlMap = new Map();
    
    sources.forEach(source => {
        // Determine the primary URL for this source
        const primaryUrl = source.google_drive_url || source.source_url || source.url || source.title || 'unknown';
        
        if (urlMap.has(primaryUrl)) {
            // If we already have this URL, merge the information
            const existing = urlMap.get(primaryUrl);
            
            // Keep the highest confidence score
            if ((source.confidence || 0) > (existing.confidence || 0)) {
                existing.confidence = source.confidence;
            }
            
            // Combine content if different
            if (source.content && source.content !== existing.content) {
                if (existing.content) {
                    existing.content = existing.content + '\n\n...\n\n' + source.content;
                } else {
                    existing.content = source.content;
                }
            }
            
            // Combine page numbers if different
            if (source.page_number && source.page_number !== existing.page_number) {
                if (existing.page_number) {
                    existing.page_number = existing.page_number + ', ' + source.page_number;
                } else {
                    existing.page_number = source.page_number;
                }
            }
            
            // Combine section titles if different
            if (source.section_title && source.section_title !== existing.section_title) {
                if (existing.section_title) {
                    existing.section_title = existing.section_title + ', ' + source.section_title;
                } else {
                    existing.section_title = source.section_title;
                }
            }
            
            // Increment duplicate count
            existing.duplicateCount = (existing.duplicateCount || 1) + 1;
            
        } else {
            // First time seeing this URL, add it to the map
            const sourceCopy = { ...source };
            sourceCopy.duplicateCount = 1;
            urlMap.set(primaryUrl, sourceCopy);
        }
    });
    
    // Convert map values back to array
    return Array.from(urlMap.values());
}

// Export functions for potential use elsewhere
window.DocumentRAG = {
    submitQuery,
    clearConversation,
    addMessageToConversation,
    conversationHistory
}; 