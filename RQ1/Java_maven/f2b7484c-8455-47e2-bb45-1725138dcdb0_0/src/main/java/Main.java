import org.springframework.transaction.annotation.Transactional;
import javax.persistence.Entity;
@Transactional
public class Main {
    public void saveEntity(Entity entity) {
        // Repository save operation
        repository.save(entity);
    }
}