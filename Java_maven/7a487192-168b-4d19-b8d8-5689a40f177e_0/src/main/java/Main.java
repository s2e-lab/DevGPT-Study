public class Main {
   public static void main(string [] args) {
      Mono<String> result = webClient.get()
      .uri("/api/resource")
      .retrieve()
      .bodyToMono(String.class);

      // To get the result synchronously
      String response = result.block();
    }
}