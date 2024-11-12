package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv" // Import godotenv
	"google.golang.org/api/option"
)

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
	err := godotenv.Load()
	if err != nil {
		log.Println("Error loading .env file")
	}

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatalln("Environment variable GEMINI_API_KEY not set")
	}

	ctx := context.Background() // Declare ctx here

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("Error creating client: %v", err)
	}
	defer client.Close()

	r := gin.Default()

	r.POST("/classify", func(c *gin.Context) {
		file, err := c.FormFile("file")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "No file part"})
			return
		}

		filePath := "/tmp/captured_image.jpg"
		if err := c.SaveUploadedFile(file, filePath); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file"})
			return
		}

		classification, err := classifyImage(ctx, client, filePath)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"category": classification})
		defer os.Remove(filePath)
	})

	r.Run(":5000")
}
