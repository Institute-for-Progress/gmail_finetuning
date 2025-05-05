# Gmail Fine-tuning System

This system allows you to fine-tune language models for email response generation using your Gmail data.

## Prerequisites

- Python 3.8 or higher
- Git
- OpenAI API key
- Gmail Takeout data (MBOX format)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd gmail-finetuning
```

2. Run the setup script:
```bash
./setup.sh
```
This will:
- Check Python version
- Create a virtual environment
- Install dependencies
- Set up directory structure
- Create configuration files

3. Configure your environment:
   - Edit `.env` and add your OpenAI API key
   - **Copy `config/params.env.example` to `config/params.env` and edit as needed.**
     - The `.example` file is provided as a template for configuration. It does not contain any sensitive information, but allows each user to customize their own settings without affecting others or risking accidental overwrites.
   - The default configuration in `config/params.env` uses GPT-4.1
   - You can modify model parameters in `config/params.env`

4. Prepare your Gmail data:
   a. Go to [Google Takeout](https://takeout.google.com/)
   b. Select only "Mail"
   c. Choose export format: MBOX
   d. Download your data
   e. Extract the downloaded zip file
   f. Place the extracted "Takeout" folder in the `data/` directory
   g. The final path should be: `data/Takeout/Mail/...`

## Usage

### Training the Model

1. Test the system:
```bash
./run.sh --test
```
This will:
- Validate your directory structure
- Test your API connection
- Generate sample training data
- Test the base model with different email types

2. Run the full training process:
```bash
./run.sh
```
This will:
- Process your Gmail data
- Create training examples
- Fine-tune the model
- Save the model ID in `config/params.env`

### Using the Fine-tuned Model

After training is complete, you can use the model to generate email responses:

```bash
./generate_email_response.py
```

This will start an interactive session where you can:
1. Type an email (press Enter twice when done)
2. Get an AI-generated response
3. Type 'quit' to exit

## Configuration

### Model Options
You can choose between different models by uncommenting the desired configuration in `config/params.env`:

- GPT-3.5 Turbo: Fast, good for testing
- GPT-4 Turbo: Better quality, more expensive
- GPT-4.1: Latest model (default)

> **Note:**
> The code checks for the *full model ID* (not just the base name) when verifying model availability. This is important for fine-tuned models, whose IDs are longer (e.g., `ft:gpt-4.1-2025-04-14:...`). For base models, the full ID is just the base name (e.g., `gpt-4.1-2025-04-14`). This logic works for both base and fine-tuned models.
>
> **Testing with Fine-tuned vs. Base Model:**
> By default, if you have a value set for `TRAINED_MODEL_ID` in `config/params.env`, the system will use your fine-tuned model for testing and inference. If you want to test or use the base model instead, simply remove or comment out the `TRAINED_MODEL_ID` line in your configuration file.
Example configuration:
```env
MODEL_NAME=gpt-4.1-2025-04-14
MODEL_SUFFIX=email_tuned
MAX_TOKENS=300
TEMPERATURE=0.7
```

### Environment Variables
In your `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
```

### System Prompt
You can customize the email assistant's behavior by modifying the `SYSTEM_PROMPT` in `config/params.env`. The default prompt instructs the model to:
1. Maintain professional tone
2. Address all points
3. Keep responses concise
4. Include appropriate greetings/closings
5. Format properly as email

## Troubleshooting

1. **Setup Issues**:
   - If `setup.sh` fails, check your Python version: `python3 --version`
   - Make sure you have write permissions in the directory
   - Check if all required files are created

2. **Directory Structure**:
   ```
   gmail-finetuning/
   ├── data/
   │   └── Takeout/
   │       └── Mail/
   │           └── ...mbox files...
   ├── config/
   │   └── params.env
   ├── .env
   └── venv/
   ```

3. **Common Errors**:
   - "No virtual environment found": Run `./setup.sh`
   - "API key not found": Check your `.env` file
   - "Directory not found": Run `./setup.sh` to create directories
   - "Permission denied": Make scripts executable with `chmod +x *.sh`
   - "No trained model ID found": Run the training process first

4. **API Errors**:
   - Verify your API key in `.env`
   - Check your OpenAI account for sufficient credits
   - Ensure you have access to the selected model

## Requirements

- Python 3.8+
- OpenAI API key
- Gmail Takeout data (MBOX format)
- Sufficient OpenAI credits for fine-tuning

This codebase was (mostly AI) generated in a single afternoon-- so absolutely no promises as to how well it will perform.