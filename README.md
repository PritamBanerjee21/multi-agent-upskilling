# AI-Powered Career Assistant

An intelligent web application that analyzes resumes, provides personalized career suggestions, calculates ATS scores, and offers interactive career guidance through a conversational AI interface.

## Core Components

### Resume Processing System
- Multiple file format support (PDF, DOCX, DOC, JPG, PNG)
- Text extraction and parsing
- Vector embedding generation
- FAISS vector store integration

### LangGraph Agent
- Conversational state management
- Tool integration and orchestration
- Memory management
- Context-aware responses
- Real-time streaming support

### Career Analysis Engine
- Skill gap identification
- Industry trend analysis
- Learning resource recommendations
- Personalized career path suggestions

### ATS Score Calculator
- Resume-job description matching
- Keyword extraction and analysis
- Skills compatibility scoring
- Improvement recommendations

## API Documentation

### File Upload Endpoints
- `POST /handle_upload`
  - Accepts multipart form data
  - Supports PDF, DOCX, DOC, JPG, PNG
  - Returns success/error status

### Chat Endpoints
- `POST /chat`
  - Accepts JSON with message
  - Returns AI response with markdown formatting
  - Maintains session context

### ATS Score Endpoints
- `POST /calculate_ats_score`
  - Accepts job description
  - Returns matching score and analysis
  - Provides keyword matching details

### Suggestion Endpoints
- `GET /stream_suggestions`
  - Server-sent events
  - Real-time career suggestions
  - Learning resource recommendations

## Development Guidelines

### Setting Up Development Environment
1. Clone the repository
2. Create virtual environment
3. Install dependencies
4. Set up environment variables
5. Run tests and verify setup

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes
- Write meaningful commit messages

### Testing
- Unit tests for core functionality
- Integration tests for API endpoints
- Test coverage monitoring
- Performance testing for vector operations

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Troubleshooting

### Common Issues
1. File upload errors
   - Check file permissions
   - Verify supported formats
   - Check file size limits

2. API connection issues
   - Verify API keys
   - Check network connectivity
   - Validate request formats

3. Vector store problems
   - Check memory usage
   - Verify embedding generation
   - Monitor cache size

## Performance Optimization

- Implement caching for frequent queries
- Optimize vector store operations
- Use lazy loading for large files
- Implement request batching
- Monitor memory usage

## Security Considerations

- Input validation
- File type verification
- API key protection
- Session management
- Rate limiting
- Secure file handling


## Contact

Your Name - your.email@example.com
Project Link: [Click Here](https://github.com/sohamfcb/project-name)

## Acknowledgments

- LangChain team for the excellent framework
- Google for Gemini API
- Groq team for their API
- Flask community for the web framework
- All other open-source contributors