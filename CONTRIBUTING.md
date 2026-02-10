# Contributing to Gold Trading AI

Thank you for your interest in contributing to Gold Trading AI!

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs
- Include detailed reproduction steps
- Attach relevant logs or screenshots

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guidelines
- Write docstrings for all functions
- Add type hints where possible
- Include unit tests for new features

### Commit Message Format
We use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

## Development Setup

```bash
# Clone the repository
git clone https://github.com/IlyasFardaouix/GOLD-TRADING-AI.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py
```

