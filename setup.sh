#!/bin/bash

# RAG-Anything FastAPI Setup Script
echo "🚀 RAG-Anything FastAPI Setup"

# Check if .env exists, if not copy from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "✅ .env file created. Please edit it with your OpenAI API key."
    echo "   Required: OPENAI_API_KEY=your_openai_api_key_here"
else
    echo "✅ .env file already exists"
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

echo "✅ Docker and docker-compose are installed"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads rag_storage input output

echo "✅ Directories created"

# Build and start the application
echo "🔨 Building and starting the application..."
docker-compose up --build -d

echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is healthy
echo "🏥 Checking application health..."
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ Application is running successfully!"
    echo ""
    echo "🌐 API Documentation: http://localhost:8000/docs"
    echo "🔍 Health Check: http://localhost:8000/health"
    echo ""
    echo "📚 Next steps:"
    echo "1. Upload a document via the API or web interface"
    echo "2. Query your documents using the query endpoints"
    echo ""
    echo "📖 For more information, see README.md"
else
    echo "⚠️  Application may still be starting up. Check logs with: docker-compose logs -f"
fi

# Show logs command
echo ""
echo "📋 Useful commands:"
echo "  View logs: docker-compose logs -f rag-api"
echo "  Stop app:  docker-compose down"
echo "  Restart:   docker-compose restart"
