public class Main {
   public static void main(String [] args) {
        MyObject myObject = new MyObject();
        Mono<MyObject> objectMono = Mono.just(myObject);

        webClient.post()
            .uri("/api/resource")
            .contentType(MediaType.APPLICATION_JSON)
            .body(objectMono, MyObject.class)
            .retrieve()
            .bodyToMono(MyObject.class)
            .subscribe(response -> System.out.println(response));
   }
}