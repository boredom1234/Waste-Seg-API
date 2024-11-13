package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/gin-gonic/gin"
	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv" // Import godotenv
	"google.golang.org/api/option"
)

// Function to classify image
func classifyImage(ctx context.Context, client *genai.Client, filePath string) (string, error) {
	fileURI := uploadToGemini(ctx, client, filePath, "image/jpeg")

	model := client.GenerativeModel("gemini-1.5-pro-002")

	session := model.StartChat()
	session.History = []*genai.Content{
		{
			Role: "user",
			Parts: []genai.Part{
				genai.FileData{URI: fileURI},
				genai.Text("Classify this image as 'metal', 'clothes', 'paper','plastic'. DONT GIVE ANYTHING ELSE AS ANSWER."),
			},
		},
	}

	resp, err := session.SendMessage(ctx, genai.Text("Classify this image as 'metal', 'clothes', 'paper','plastic'. DONT GIVE ANYTHING ELSE AS ANSWER."))
	if err != nil {
		return "", fmt.Errorf("error sending message: %w", err)
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return "", fmt.Errorf("empty response received")
	}

	var classification string
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			classification += string(text)
		}
	}

	return classification, nil
}

// Function to upload file to Gemini
func uploadToGemini(ctx context.Context, client *genai.Client, path, mimeType string) string {
	file, err := os.Open(path)
	if err != nil {
		log.Fatalf("Error opening file: %v", err)
	}
	defer file.Close()

	options := genai.UploadFileOptions{
		DisplayName: path,
		MIMEType:    mimeType,
	}
	fileData, err := client.UploadFile(ctx, "", file, &options)
	if err != nil {
		log.Fatalf("Error uploading file: %v", err)
	}

	return fileData.URI
}

func main() {
	// Load environment variables
	err := godotenv.Load()
	if err != nil {
		log.Println("Error loading .env file")
	}

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatalln("Environment variable GEMINI_API_KEY not set")
	}

	// Initialize context and client
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("Error creating client: %v", err)
	}
	defer client.Close()

	// Setup Gin router
	r := gin.Default()

	// POST endpoint for ESP32 to upload image and classify
	r.POST("/classify", func(c *gin.Context) {
		file, err := c.FormFile("file")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "No file part"})
			return
		}

		// Save file to a temporary location
		tempFilePath := filepath.Join(os.TempDir(), "captured_image.jpg")
		if err := c.SaveUploadedFile(file, tempFilePath); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file"})
			return
		}

		// Classify the uploaded image
		classification, err := classifyImage(ctx, client, tempFilePath)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Send back classification result
		c.JSON(http.StatusOK, gin.H{"category": classification})

		// Clean up
		defer os.Remove(tempFilePath)
	})

	// Simple health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	// Start the server on port 5000
	r.Run(":5000")
}
