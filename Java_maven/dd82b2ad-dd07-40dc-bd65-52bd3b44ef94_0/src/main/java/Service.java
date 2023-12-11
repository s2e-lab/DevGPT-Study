import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;

interface MyService {
    void doSomething();
}

@Service
@ConditionalOnProperty(name = "my.service.type", havingValue = "serviceA")
class ServiceA implements MyService {
    @Override
    public void doSomething() {
        // Implementation for Service A
    }
}

@Service
@ConditionalOnProperty(name = "my.service.type", havingValue = "serviceB")
class ServiceB implements MyService {
    @Override
    public void doSomething() {
        // Implementation for Service B
    }
}
