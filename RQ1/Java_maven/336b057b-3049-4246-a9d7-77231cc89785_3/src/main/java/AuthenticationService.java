public class AuthenticationService {
    private UserRepository userRepository;
    private JwtUtils jwtUtils;

    public AuthenticationService(UserRepository userRepository, JwtUtils jwtUtils) {
        this.userRepository = userRepository;
        this.jwtUtils = jwtUtils;
    }

    public String authenticate(String username, String password) {
        Optional<User> optionalUser = userRepository.findByUsername(username);
        if (optionalUser.isPresent()) {
            User user = optionalUser.get();
            if (user.getPassword().equals(password)) {
                return jwtUtils.createJwtToken(user);
            }
        }
        return null;
    }

    public boolean isTokenValid(String token) {
        return jwtUtils.verifyJwtToken(token);
    }

    public String extractUserIdFromToken(String token) {
        return jwtUtils.extractUserIdFromToken(token);
    }
}
