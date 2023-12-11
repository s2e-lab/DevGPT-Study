import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
public class SnackController {
    @Autowired
    private SnackService snackService;

    @GetMapping("/search")
    public List<SearchDTO> searchSnacks(@RequestParam String keyword, @RequestParam String[] filter) {
        List<SearchDTO> filteredSnacks = snackService.searchSnacksWithFilter(keyword, filter);
        return filteredSnacks;
    }
}
