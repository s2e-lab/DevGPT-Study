public interface MyService {
    void doSomething();
}

@Service
@ConditionalOnProperty(name = "my.service.type", havingValue = "serviceA")
public class ServiceA implements MyService {
    @Override
    public void doSomething() {
        // Implementation for Service A
    }
}

@Service
@ConditionalOnProperty(name = "my.service.type", havingValue = "serviceB")
public class ServiceB implements MyService {
    @Override
    public void doSomething() {
        // Implementation for Service B
    }
}
