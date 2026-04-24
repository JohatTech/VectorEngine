import os
import base64
import requests

class EmailNotifier:
    def __init__(self):
        self.email_url = os.getenv("EMAIL_API_TOKEN")

    def send_report(self, file_path: str, title: str, message: str, correo: str):
        print("ENVIANDO ARCHIVO POR CORREO")
        
        if not self.email_url:
            print("Email URL is not configured. Skipping email sent.")
            return

        try:
            with open(file_path, "rb") as file:
                file_content = base64.b64encode(file.read()).decode("utf-8")
                
            response = requests.post(
                self.email_url,
                json={
                    "fileName": file_path, 
                    "fileContent": file_content, 
                    "Asunto": title,
                    "Message": message, 
                    "correo": correo
                },
                timeout=30
            )
            print(f"Email sent response: {response.status_code}")
            return response
        except requests.RequestException as e:
            print(f"Error sending the email: {e}")
            return None
