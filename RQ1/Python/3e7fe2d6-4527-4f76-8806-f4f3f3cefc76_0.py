import wtforms as wtf
import flask_wtf as fwtf
import game_resources as gr

class M_MapAgent(type(fwtf.FlaskForm), type):
    def __new__(cls, name, bases, attrs):
        map_agent_attr = {}
        for map_enum in gr.Map:
            field_name = map_enum.name.lower()
            field = wtf.SelectField(map_enum.name, choices=[agent.name for agent in gr.Agent], validators=[wtf.validators.Optional()])
            map_agent_attr[field_name] = field
        return super().__new__(cls, name, bases, attrs | map_agent_attr)

class MapAgent(fwtf.FlaskForm, metaclass=M_MapAgent):
    name = wtf.StringField('Name', validators=[wtf.validators.DataRequired()])
    game_mode = wtf.SelectField('Game Mode', choices=[mode.name for mode in gr.GameMode], validators=[wtf.validators.DataRequired()])
    submit = wtf.SubmitField('Submit_label')
