@RestController
@RequestMapping("/api")
@CrossOrigin("http://localhost:5173/")
public class ApiController {
    // Other code in the class...

    @GetMapping("/check-login")
    public Map<String, Boolean> checkLogin(@RequestHeader("Authorization") String token) {
        Map<String, Boolean> response = new HashMap<>();
        response.put("isLoggedIn", authenticationService.isTokenValid(token));
        return response;
    }

    // Other code in the class...
}
