public class Main {
   public static void main(string [] args) {
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