import org.springframework.stereotype.Service;

@Service
public class MyServiceUser {
    private final MyService myService;

    public MyServiceUser(MyService myService) {
        this.myService = myService;
    }

    // Use the myService instance as needed
}

interface MyService {
    void doSomething();
}
