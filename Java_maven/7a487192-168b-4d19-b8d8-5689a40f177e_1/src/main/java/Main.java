public class Main {
   public static void main(String [] args) {
      Mono<String> result = webClient.get()
      .uri("/api/resource")
      .retrieve()
      .bodyToMono(String.class);

      // To get the result synchronously (not recommended in a reactive context)
      String response = result.block();
   }
}