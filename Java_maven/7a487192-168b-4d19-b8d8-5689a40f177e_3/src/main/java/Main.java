import org.springframework.http.HttpStatus;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

class CustomException extends RuntimeException {
    public CustomException(String message) {
        super(message);
    }
}

public class Main {
    public static void main(String[] args) {
        WebClient webClient = WebClient.create("http://your-api-base-url");

        Mono<String> result = webClient.get()
                .uri("/api/resource")
                .retrieve()
                .onStatus(HttpStatus::is4xxClientError, clientResponse ->
                        Mono.error(new CustomException("Client Error!")))
                .onStatus(HttpStatus::is5xxServerError, clientResponse ->
                        Mono.error(new CustomException("Server Error!")))
                .bodyToMono(String.class);
    }
}
