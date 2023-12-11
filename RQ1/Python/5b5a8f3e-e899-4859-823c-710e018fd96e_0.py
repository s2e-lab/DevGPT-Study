class ProjectTask(models.Model):
    _inherit = 'project.task'

    sub_tasks_count = fields.Integer(compute='_compute_sub_tasks_count')

    @api.depends('sub_tasks')
    def _compute_sub_tasks_count(self):
        for task in self:
            task.sub_tasks_count = len(task.sub_tasks)
