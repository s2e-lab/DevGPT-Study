@SpringBootApplication //(exclude = SecurityAutoConfiguration.class) // Spring Security 인증 기능 제외
public class SpringAuthApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringAuthApplication.class, args);
    }
}
